import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

# ============================================================
# 0. REPRODUCIBILITY
# ============================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# 1. LOAD THE TRAINING KG
# ============================================================

triplets = pd.read_csv("data/triplets_train.csv")
triplets.columns = ["head", "relation", "tail"]

print("Number of training KG triplets:", len(triplets))
print(triplets.head())

# ============================================================
# 2. CREATE ENTITY AND RELATION MAPPINGS
# ============================================================

entity_to_id = {}
relation_to_id = {}

for _, row in triplets.iterrows():

    head = row["head"]
    relation = row["relation"]
    tail = row["tail"]

    if head not in entity_to_id:
        entity_to_id[head] = len(entity_to_id)

    if tail not in entity_to_id:
        entity_to_id[tail] = len(entity_to_id)

    if relation not in relation_to_id:
        relation_to_id[relation] = len(relation_to_id)

num_entities = len(entity_to_id)
num_relations = len(relation_to_id)

id_to_entity = {v: k for k, v in entity_to_id.items()}
id_to_relation = {v: k for k, v in relation_to_id.items()}

print("\nNumber of entities:", num_entities)
print("Number of relations:", num_relations)

# ============================================================
# 3. CONVERT TRIPLETS TO INTEGER IDs
# ============================================================

triplet_indices = [
    (
        entity_to_id[row["head"]],
        relation_to_id[row["relation"]],
        entity_to_id[row["tail"]]
    )
    for _, row in triplets.iterrows()
]

positive_triplets = set(triplet_indices)

# ============================================================
# 4. ENTITY TYPE POOLS (for type-constrained negative sampling)
# ============================================================

entity_type_pool = defaultdict(list)

for entity_str, eid in entity_to_id.items():
    entity_type = entity_str.split("_")[0]
    entity_type_pool[entity_type].append(eid)

entity_type_pool = {
    entity_type: np.array(ids, dtype=np.int64)
    for entity_type, ids in entity_type_pool.items()
}

print("\nEntity type pools:")
for entity_type, ids in entity_type_pool.items():
    print(f"  {entity_type}: {len(ids)} entities")


def get_entity_type(entity_id):
    return id_to_entity[entity_id].split("_")[0]


# ============================================================
# 5. BERNOULLI HEAD/TAIL CORRUPTION PROBABILITIES
# ============================================================

# For each relation, compute:
#   tph = average number of tails per distinct head  (1-to-many-ness)
#   hpt = average number of heads per distinct tail  (many-to-1-ness)
#
# Then P(corrupt head) = tph / (tph + hpt).
#
# Intuition: if a relation is 1-to-many (high tph, e.g. one user
# "Likes" many movies), corrupting the HEAD introduces far fewer
# accidental false negatives than corrupting the TAIL (since many
# tails are legitimately valid for a given head). The Bernoulli
# trick (Wang et al., TransH) biases corruption toward whichever
# side is less likely to create a false negative.

relation_head_tails = defaultdict(lambda: defaultdict(set))
relation_tail_heads = defaultdict(lambda: defaultdict(set))

for h, r, t in triplet_indices:
    relation_head_tails[r][h].add(t)
    relation_tail_heads[r][t].add(h)

relation_corrupt_head_prob = {}

for r in range(num_relations):

    heads_dict = relation_head_tails[r]
    tails_dict = relation_tail_heads[r]

    if len(heads_dict) == 0 or len(tails_dict) == 0:
        relation_corrupt_head_prob[r] = 0.5
        continue

    tph = np.mean([len(v) for v in heads_dict.values()])
    hpt = np.mean([len(v) for v in tails_dict.values()])

    relation_corrupt_head_prob[r] = float(tph / (tph + hpt))

print("\nBernoulli corrupt-head probability per relation:")
for r in range(num_relations):
    print(f"  {id_to_relation[r]}: P(corrupt head) = {relation_corrupt_head_prob[r]:.3f}")


# ============================================================
# 6. TRAIN / VALIDATION SPLIT (for link-prediction validation)
# ============================================================

np.random.shuffle(triplet_indices)

val_fraction = 0.05
split_point = int(len(triplet_indices) * (1 - val_fraction))

train_triplet_indices = triplet_indices[:split_point]
val_triplet_indices = triplet_indices[split_point:]

print(f"\nTrain triplets: {len(train_triplet_indices)}")
print(f"Validation triplets: {len(val_triplet_indices)}")

# ============================================================
# 7. TRANSR MODEL
# ============================================================

class TransR(nn.Module):

    def __init__(self, num_entities, num_relations, entity_dim=100, relation_dim=100):

        super(TransR, self).__init__()

        self.entity_embeddings = nn.Embedding(num_entities, entity_dim)
        self.relation_embeddings = nn.Embedding(num_relations, relation_dim)

        self.projection_matrices = nn.Parameter(
            torch.empty(num_relations, entity_dim, relation_dim)
        )

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        nn.init.xavier_uniform_(self.projection_matrices)

        with torch.no_grad():
            self.entity_embeddings.weight.data = nn.functional.normalize(
                self.entity_embeddings.weight.data, p=2, dim=1
            )

    def forward(self, h, r, t):

        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)

        proj_h_emb = torch.bmm(
            h_emb.unsqueeze(1), self.projection_matrices[r]
        ).squeeze(1)

        proj_t_emb = torch.bmm(
            t_emb.unsqueeze(1), self.projection_matrices[r]
        ).squeeze(1)

        return proj_h_emb, r_emb, proj_t_emb

    def score_function(self, h_emb, r_emb, t_emb):
        # Lower distance = more plausible triple
        return torch.norm(h_emb + r_emb - t_emb, p=1, dim=1)


# ============================================================
# 8. HYPERPARAMETERS
# ============================================================

ENTITY_DIM = 100
RELATION_DIM = 100

MARGIN = 2.0

LEARNING_RATE = 0.001
NUM_EPOCHS = 60
BATCH_SIZE = 256
WEIGHT_DECAY = 1e-4

# Number of negative triples generated per positive triple.
NEGATIVES_PER_POSITIVE = 4

# Self-adversarial weighting temperature (RotatE-style). Lower
# temperature -> sharper focus on the hardest negatives (the
# ones the current model already scores as fairly plausible).
# alpha = 0 disables adversarial weighting (uniform average,
# equivalent to the previous version).
ADVERSARIAL_ALPHA = 1.0

EVAL_EVERY = 5
PRINT_EVERY = 1

NUM_EVAL_CORRUPTIONS = 50
HITS_K = 10
VAL_SAMPLE_SIZE = 500

# ============================================================
# 9. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDevice:", device)

if device.type == "cpu":
    print(
        "WARNING: running on CPU. Use a GPU runtime for a large speedup."
    )

# ============================================================
# 10. INITIALIZE TRANSR
# ============================================================

transR = TransR(
    num_entities=num_entities,
    num_relations=num_relations,
    entity_dim=ENTITY_DIM,
    relation_dim=RELATION_DIM
).to(device)

optimizer = optim.AdamW(
    transR.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# Cosine annealing lets the LR decay smoothly across the full
# run instead of staying flat, helping the model settle more
# precisely by the end of training.
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS
)

# ============================================================
# 11. TYPE-CONSTRAINED + BERNOULLI NEGATIVE SAMPLING
# ============================================================

MAX_TYPE_CONSTRAINED_ATTEMPTS = 20


def generate_negative_samples(
    positive_batch,
    entity_type_pool,
    positive_triplets,
    relation_corrupt_head_prob,
    negatives_per_positive=1,
    num_entities=None
):
    """
    Same type-constrained corruption + fallback as before, but
    now the head-vs-tail corruption choice is drawn from the
    relation-specific Bernoulli probability instead of a flat
    50/50 coin flip, reducing accidental false negatives for
    1-to-many / many-to-1 relations.
    """

    negative_triplets = []

    for h, r, t in positive_batch:

        p_corrupt_head = relation_corrupt_head_prob.get(r, 0.5)

        for _ in range(negatives_per_positive):

            candidate = None
            corrupt_head = np.random.rand() < p_corrupt_head

            for _ in range(MAX_TYPE_CONSTRAINED_ATTEMPTS):

                if corrupt_head:
                    h_type = get_entity_type(h)
                    new_h = int(np.random.choice(entity_type_pool[h_type]))
                    attempt = (new_h, r, t)
                else:
                    t_type = get_entity_type(t)
                    new_t = int(np.random.choice(entity_type_pool[t_type]))
                    attempt = (h, r, new_t)

                if attempt not in positive_triplets:
                    candidate = attempt
                    break

            if candidate is None:
                for _ in range(MAX_TYPE_CONSTRAINED_ATTEMPTS):
                    if corrupt_head:
                        new_h = np.random.randint(0, num_entities)
                        attempt = (new_h, r, t)
                    else:
                        new_t = np.random.randint(0, num_entities)
                        attempt = (h, r, new_t)

                    if attempt not in positive_triplets:
                        candidate = attempt
                        break

            if candidate is None:
                candidate = attempt

            negative_triplets.append(candidate)

    return negative_triplets


# ============================================================
# 12. VALIDATION METRIC: FILTERED MRR / HITS@K
# ============================================================

def evaluate_mrr_hits(
    model,
    val_triplets,
    entity_type_pool,
    positive_triplets,
    k=10,
    num_corruptions=50,
    sample_size=500
):
    model.eval()

    sample = val_triplets
    if len(val_triplets) > sample_size:
        idx = np.random.choice(len(val_triplets), size=sample_size, replace=False)
        sample = [val_triplets[i] for i in idx]

    ranks = []

    with torch.no_grad():

        for h, r, t in sample:

            t_type = get_entity_type(t)
            pool = entity_type_pool[t_type]

            if len(pool) <= 1:
                continue

            candidates = np.random.choice(pool, size=num_corruptions, replace=True)
            candidates = np.append(candidates, t)

            h_batch = torch.full((len(candidates),), h, dtype=torch.long, device=device)
            r_batch = torch.full((len(candidates),), r, dtype=torch.long, device=device)
            c_batch = torch.tensor(candidates, dtype=torch.long, device=device)

            h_emb, r_emb, c_emb = model(h_batch, r_batch, c_batch)
            scores = model.score_function(h_emb, r_emb, c_emb)

            true_score = scores[-1]
            rank = int((scores[:-1] < true_score).sum().item()) + 1
            ranks.append(rank)

    model.train()

    if len(ranks) == 0:
        return float("nan"), float("nan")

    ranks = np.array(ranks)
    mrr = float(np.mean(1.0 / ranks))
    hits_at_k = float(np.mean(ranks <= k))

    return mrr, hits_at_k


# ============================================================
# 13. TRAIN TRANSR
# ============================================================

print("\n========================================")
print("TRAINING TRANSR")
print("========================================")

best_mrr = -np.inf
best_model_state = None

for epoch in range(NUM_EPOCHS):

    np.random.shuffle(train_triplet_indices)

    epoch_loss = 0.0
    num_batches = 0

    batch_starts = range(0, len(train_triplet_indices), BATCH_SIZE)

    for start in batch_starts:

        batch = train_triplet_indices[start:start + BATCH_SIZE]

        # ----------------------------------------------------
        # Positive triples
        # ----------------------------------------------------

        heads, relations, tails = zip(*batch)

        heads = torch.tensor(heads, dtype=torch.long, device=device)
        relations = torch.tensor(relations, dtype=torch.long, device=device)
        tails = torch.tensor(tails, dtype=torch.long, device=device)

        # ----------------------------------------------------
        # Negative triples (type-constrained + Bernoulli)
        # ----------------------------------------------------

        negative_batch = generate_negative_samples(
            batch,
            entity_type_pool,
            positive_triplets,
            relation_corrupt_head_prob,
            negatives_per_positive=NEGATIVES_PER_POSITIVE,
            num_entities=num_entities
        )

        neg_heads, neg_relations, neg_tails = zip(*negative_batch)

        neg_heads = torch.tensor(neg_heads, dtype=torch.long, device=device)
        neg_relations = torch.tensor(neg_relations, dtype=torch.long, device=device)
        neg_tails = torch.tensor(neg_tails, dtype=torch.long, device=device)

        # ----------------------------------------------------
        # Forward pass (positives)
        # ----------------------------------------------------

        pos_h, pos_r, pos_t = transR(heads, relations, tails)
        pos_scores = transR.score_function(pos_h, pos_r, pos_t)  # (batch,)

        # ----------------------------------------------------
        # Forward pass (negatives)
        # ----------------------------------------------------

        neg_h, neg_r, neg_t = transR(neg_heads, neg_relations, neg_tails)
        neg_scores_flat = transR.score_function(neg_h, neg_r, neg_t)  # (batch * K,)

        # Reshape to (batch, K) -- negatives were generated in
        # order: for each positive, K consecutive negatives.
        neg_scores = neg_scores_flat.view(len(batch), NEGATIVES_PER_POSITIVE)

        # ----------------------------------------------------
        # Self-adversarial weighting (RotatE-style)
        #
        # Lower score = more plausible triple, so the "hardest"
        # negatives (the ones the model currently finds most
        # plausible) have the LOWEST score. We give them the
        # highest weight so the gradient focuses on the negatives
        # that are actually still hard to distinguish, instead of
        # spending equal effort on negatives the model already
        # rejects easily.
        #
        # Weights are computed with no_grad -- they only reweight
        # the loss, they are not themselves optimized.
        # ----------------------------------------------------

        with torch.no_grad():
            adversarial_weights = torch.softmax(
                -ADVERSARIAL_ALPHA * neg_scores, dim=1
            )  # (batch, K)

        # ----------------------------------------------------
        # Weighted margin ranking loss
        # ----------------------------------------------------

        pos_scores_expanded = pos_scores.unsqueeze(1)  # (batch, 1)

        per_negative_loss = torch.relu(
            pos_scores_expanded - neg_scores + MARGIN
        )  # (batch, K)

        loss = torch.mean(
            torch.sum(adversarial_weights * per_negative_loss, dim=1)
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            transR.entity_embeddings.weight.data = nn.functional.normalize(
                transR.entity_embeddings.weight.data, p=2, dim=1
            )

        epoch_loss += loss.item()
        num_batches += 1

    average_loss = epoch_loss / num_batches

    lr_scheduler.step()

    if (epoch + 1) % PRINT_EVERY == 0 or (epoch + 1) == NUM_EPOCHS:
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} - "
            f"Average Loss: {average_loss:.6f} - LR: {current_lr:.6f}",
            flush=True
        )

    if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == NUM_EPOCHS:

        mrr, hits_at_k = evaluate_mrr_hits(
            transR,
            val_triplet_indices,
            entity_type_pool,
            positive_triplets,
            k=HITS_K,
            num_corruptions=NUM_EVAL_CORRUPTIONS,
            sample_size=VAL_SAMPLE_SIZE
        )

        print(f"  -> Validation MRR: {mrr:.4f} | Hits@{HITS_K}: {hits_at_k:.4f}")

        if mrr > best_mrr:
            best_mrr = mrr
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in transR.state_dict().items()
            }
            print("  -> Best validation MRR updated.")

# ============================================================
# 14. RESTORE BEST MODEL (by validation MRR)
# ============================================================

if best_model_state is not None:
    transR.load_state_dict(best_model_state)
    transR = transR.to(device)
    print(f"\nRestored best model with validation MRR: {best_mrr:.4f}")

# ============================================================
# 15. EXTRACT TRAINED EMBEDDINGS
# ============================================================

entity_embeddings_movie = transR.entity_embeddings.weight.detach().cpu().numpy()
relation_embeddings_movie = transR.relation_embeddings.weight.detach().cpu().numpy()

# ============================================================
# 16. SAVE EMBEDDINGS
# ============================================================

np.save("entity_embeddings_movie_train.npy", entity_embeddings_movie)
np.save("relation_embeddings_movie_train.npy", relation_embeddings_movie)

# ============================================================
# 17. SAVE THE MAPPINGS
# ============================================================

entity_mapping_df = pd.DataFrame(
    [(entity_id, entity) for entity_id, entity in id_to_entity.items()],
    columns=["ID", "Entity"]
)

relation_mapping_df = pd.DataFrame(
    [(relation_id, relation) for relation_id, relation in id_to_relation.items()],
    columns=["ID", "Relation"]
)

entity_mapping_df.to_csv("movie_entity_mapping.csv", index=False)
relation_mapping_df.to_csv("movie_relation_mapping.csv", index=False)

# ============================================================
# 18. SUMMARY
# ============================================================

print("\n========================================")
print("TRANSR TRAINING COMPLETE")
print("========================================")

print("Entity embedding shape:", entity_embeddings_movie.shape)
print("Relation embedding shape:", relation_embeddings_movie.shape)
print(f"Best validation MRR: {best_mrr:.4f}")

print("\nSaved files:")
print("  entity_embeddings_movie_train.npy")
print("  relation_embeddings_movie_train.npy")
print("  movie_entity_mapping.csv")
print("  movie_relation_mapping.csv")