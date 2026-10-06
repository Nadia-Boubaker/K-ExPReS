# ============================================================
# K-ExPReS - LEAKAGE-FREE MOVIELENS RECOMMENDATION EXPERIMENT
# WITHOUT EARLY STOPPING
# ============================================================

# ============================================================
# 1. IMPORTS
# ============================================================

import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix
)

from collections import defaultdict


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# 3. CONFIGURATION
# ============================================================

USER_COL = "UserID"
ITEM_COL = "MovieID"

# Number of negative samples per positive interaction
NEGATIVES_PER_POSITIVE = 1

# ------------------------------------------------------------
# Model parameters
# ------------------------------------------------------------

EMBEDDING_DIM = 100

HIDDEN_DIM = 256
NUM_LAYERS = 1

FC_DIM_1 = 128
FC_DIM_2 = 64

DROPOUT = 0.3

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

BATCH_SIZE = 256

# The model will ALWAYS train for this many epochs.
NUM_EPOCHS = 100

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

# ------------------------------------------------------------
# Classification threshold
# ------------------------------------------------------------

# Precision / Recall / F1 / Accuracy depend on this threshold.
# AUC does not depend on this threshold.
CLASSIFICATION_THRESHOLD = 0.4


# ============================================================
# 4. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# 5. CHECK REQUIRED FILES
# ============================================================

required_files = [
    "train_ratings.csv",
    "val_ratings.csv",
    "test_ratings.csv",
    "movies.csv",
    "entity_embeddings_movie_train.npy",
    "movie_entity_mapping.csv"
]

missing_files = [
    f for f in required_files
    if not os.path.exists(f)
]

if len(missing_files) > 0:

    raise FileNotFoundError(
        "\nThe following required files are missing:\n"
        + "\n".join(missing_files)
        + "\n\nPlease generate these files first."
    )

print("\nAll required files were found.")


# ============================================================
# 6. LOAD MOVIELENS INTERACTIONS
# ============================================================

train_df = pd.read_csv("data/train_ratings.csv")
val_df = pd.read_csv("data/val_ratings.csv")
test_df = pd.read_csv("data/test_ratings.csv")
movies_df = pd.read_csv("data/movies.csv")

print("\n================ DATASETS ================")

print("Train interactions:", len(train_df))
print("Validation interactions:", len(val_df))
print("Test interactions:", len(test_df))

print("\nTrain columns:")
print(train_df.columns.tolist())


# ============================================================
# 7. VERIFY REQUIRED COLUMNS
# ============================================================

for name, df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df),
    ("movies", movies_df)
]:

    if USER_COL not in df.columns and name != "movies":

        raise ValueError(
            f"{USER_COL} is missing from {name}_ratings.csv"
        )

    if ITEM_COL not in df.columns:

        raise ValueError(
            f"{ITEM_COL} is missing from {name} dataset"
        )


# ============================================================
# 8. CONVERT IDs TO INTEGER
# ============================================================

train_df[USER_COL] = train_df[USER_COL].astype(int)
val_df[USER_COL] = val_df[USER_COL].astype(int)
test_df[USER_COL] = test_df[USER_COL].astype(int)

train_df[ITEM_COL] = train_df[ITEM_COL].astype(int)
val_df[ITEM_COL] = val_df[ITEM_COL].astype(int)
test_df[ITEM_COL] = test_df[ITEM_COL].astype(int)

movies_df[ITEM_COL] = movies_df[ITEM_COL].astype(int)


# ============================================================
# 9. LOAD TRANSR EMBEDDINGS
# ============================================================

entity_embeddings = np.load(
    "entity_embeddings_movie_train.npy"
)

print("\n================ TRANSR EMBEDDINGS ================")

print(
    "Entity embedding matrix shape:",
    entity_embeddings.shape
)


# ============================================================
# 10. LOAD ENTITY MAPPING
# ============================================================

entity_mapping = pd.read_csv(
    "movie_entity_mapping.csv"
)

print("\nEntity mapping columns:")
print(entity_mapping.columns.tolist())


# ============================================================
# 11. BUILD ENTITY -> EMBEDDING ID MAPPING
# ============================================================

if "ID" not in entity_mapping.columns:

    raise ValueError(
        "movie_entity_mapping.csv must contain an 'ID' column."
    )

if "Entity" not in entity_mapping.columns:

    raise ValueError(
        "movie_entity_mapping.csv must contain an 'Entity' column."
    )

entity_to_embedding_id = {}

for _, row in entity_mapping.iterrows():

    entity = str(row["Entity"])
    entity_id = int(row["ID"])

    entity_to_embedding_id[entity] = entity_id


# ============================================================
# 12. USER / MOVIE EMBEDDING LOOKUP FUNCTIONS
# ============================================================

def get_user_entity_id(user_id):

    return f"U_{int(user_id)}"


def get_movie_entity_id(movie_id):

    return f"I_{int(movie_id)}"


def get_embedding(entity_name):

    if entity_name not in entity_to_embedding_id:

        return None

    embedding_id = entity_to_embedding_id[entity_name]

    if embedding_id < 0 or embedding_id >= len(entity_embeddings):

        return None

    return entity_embeddings[embedding_id]


# ============================================================
# 13. CHECK USER / MOVIE COVERAGE
# ============================================================

all_users = set(
    train_df[USER_COL]
    .astype(int)
    .unique()
)

all_movies = set(
    movies_df[ITEM_COL]
    .astype(int)
    .unique()
)

users_with_embeddings = 0
movies_with_embeddings = 0

for user_id in all_users:

    entity_name = get_user_entity_id(user_id)

    if entity_name in entity_to_embedding_id:

        users_with_embeddings += 1


for movie_id in all_movies:

    entity_name = get_movie_entity_id(movie_id)

    if entity_name in entity_to_embedding_id:

        movies_with_embeddings += 1


print("\n================ EMBEDDING COVERAGE ================")

print(
    f"Users with TransR embeddings: "
    f"{users_with_embeddings}/{len(all_users)}"
)

print(
    f"Movies with TransR embeddings: "
    f"{movies_with_embeddings}/{len(all_movies)}"
)


# ============================================================
# 14. BUILD KNOWN INTERACTION SET
# ============================================================

# Train + validation + test interactions are excluded
# from negative sampling so observed positive interactions
# cannot accidentally become negatives.

user_known_items = defaultdict(set)

for _, row in train_df.iterrows():

    user = int(row[USER_COL])
    item = int(row[ITEM_COL])

    user_known_items[user].add(item)


for _, row in val_df.iterrows():

    user = int(row[USER_COL])
    item = int(row[ITEM_COL])

    user_known_items[user].add(item)


for _, row in test_df.iterrows():

    user = int(row[USER_COL])
    item = int(row[ITEM_COL])

    user_known_items[user].add(item)


# ============================================================
# 15. CANDIDATE MOVIES
# ============================================================

available_movies = set()

for movie_id in all_movies:

    entity_name = get_movie_entity_id(movie_id)

    if entity_name in entity_to_embedding_id:

        available_movies.add(movie_id)


print(
    "\nMovies available for recommendation:",
    len(available_movies)
)


# ============================================================
# 16. NEGATIVE SAMPLING
# ============================================================

def generate_negative_samples(
    positive_df,
    available_items,
    user_known_items,
    negatives_per_positive=1,
    random_state=42
):

    rng = np.random.default_rng(random_state)

    records = []

    available_items_array = np.array(
        sorted(available_items),
        dtype=np.int64
    )

    for _, row in positive_df.iterrows():

        user = int(row[USER_COL])
        positive_item = int(row[ITEM_COL])

        # ----------------------------------------------------
        # Positive interaction
        # ----------------------------------------------------

        records.append({
            USER_COL: user,
            ITEM_COL: positive_item,
            "label": 1
        })

        # ----------------------------------------------------
        # Candidate negative items
        # ----------------------------------------------------

        known_items = user_known_items[user]

        candidates = np.array(
            [
                item
                for item in available_items_array
                if item not in known_items
            ],
            dtype=np.int64
        )

        if len(candidates) == 0:

            continue

        number_of_negatives = min(
            negatives_per_positive,
            len(candidates)
        )

        negative_items = rng.choice(
            candidates,
            size=number_of_negatives,
            replace=False
        )

        for negative_item in negative_items:

            records.append({
                USER_COL: user,
                ITEM_COL: int(negative_item),
                "label": 0
            })

    result = pd.DataFrame(records)

    return result


# ============================================================
# 17. GENERATE RECOMMENDATION DATASETS
# ============================================================

print("\n================ NEGATIVE SAMPLING ================")

recommendation_train = generate_negative_samples(
    train_df,
    available_movies,
    user_known_items,
    negatives_per_positive=NEGATIVES_PER_POSITIVE,
    random_state=42
)

recommendation_val = generate_negative_samples(
    val_df,
    available_movies,
    user_known_items,
    negatives_per_positive=NEGATIVES_PER_POSITIVE,
    random_state=123
)

recommendation_test = generate_negative_samples(
    test_df,
    available_movies,
    user_known_items,
    negatives_per_positive=NEGATIVES_PER_POSITIVE,
    random_state=456
)


# ============================================================
# 18. REMOVE POSSIBLE MISSING EMBEDDINGS
# ============================================================

def has_embeddings(row):

    user_entity = get_user_entity_id(
        row[USER_COL]
    )

    item_entity = get_movie_entity_id(
        row[ITEM_COL]
    )

    return (
        user_entity in entity_to_embedding_id
        and
        item_entity in entity_to_embedding_id
    )


recommendation_train = recommendation_train[
    recommendation_train.apply(has_embeddings, axis=1)
].reset_index(drop=True)

recommendation_val = recommendation_val[
    recommendation_val.apply(has_embeddings, axis=1)
].reset_index(drop=True)

recommendation_test = recommendation_test[
    recommendation_test.apply(has_embeddings, axis=1)
].reset_index(drop=True)


# ============================================================
# 19. DISPLAY DATASET BALANCE
# ============================================================

print("\nTraining labels:")
print(recommendation_train["label"].value_counts())

print("\nValidation labels:")
print(recommendation_val["label"].value_counts())

print("\nTest labels:")
print(recommendation_test["label"].value_counts())


# ============================================================
# 20. SAVE RECOMMENDATION DATASETS
# ============================================================

recommendation_train.to_csv(
    "recommendation_train.csv",
    index=False
)

recommendation_val.to_csv(
    "recommendation_val.csv",
    index=False
)

recommendation_test.to_csv(
    "recommendation_test.csv",
    index=False
)

print("\nRecommendation datasets saved.")


# ============================================================
# 21. DATASET CLASS
# ============================================================

class UserItemEmbeddingDataset(Dataset):

    def __init__(
        self,
        dataframe,
        entity_to_embedding_id,
        entity_embeddings
    ):

        self.df = dataframe.reset_index(drop=True)

        self.entity_to_embedding_id = (
            entity_to_embedding_id
        )

        self.entity_embeddings = entity_embeddings

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        user_id = int(row[USER_COL])
        item_id = int(row[ITEM_COL])

        user_entity = get_user_entity_id(
            user_id
        )

        item_entity = get_movie_entity_id(
            item_id
        )

        user_embedding_id = (
            self.entity_to_embedding_id[user_entity]
        )

        item_embedding_id = (
            self.entity_to_embedding_id[item_entity]
        )

        user_embedding = self.entity_embeddings[
            user_embedding_id
        ]

        item_embedding = self.entity_embeddings[
            item_embedding_id
        ]

        label = float(row["label"])

        return (
            torch.tensor(
                user_embedding,
                dtype=torch.float32
            ),

            torch.tensor(
                item_embedding,
                dtype=torch.float32
            ),

            torch.tensor(
                label,
                dtype=torch.float32
            )
        )


# ============================================================
# 22. CREATE PYTORCH DATASETS
# ============================================================

train_dataset = UserItemEmbeddingDataset(
    recommendation_train,
    entity_to_embedding_id,
    entity_embeddings
)

val_dataset = UserItemEmbeddingDataset(
    recommendation_val,
    entity_to_embedding_id,
    entity_embeddings
)

test_dataset = UserItemEmbeddingDataset(
    recommendation_test,
    entity_to_embedding_id,
    entity_embeddings
)


# ============================================================
# 23. CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 24. ATTENTION MODULE
# ============================================================

class Attention(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attention_layer = nn.Linear(
            hidden_dim * 2,
            1
        )

    def forward(self, lstm_output):

        scores = self.attention_layer(
            lstm_output
        ).squeeze(-1)

        attention_weights = torch.softmax(
            scores,
            dim=1
        )

        context_vector = torch.sum(
            attention_weights.unsqueeze(-1)
            * lstm_output,
            dim=1
        )

        return (
            context_vector,
            attention_weights
        )


# ============================================================
# 25. K-ExPReS RECOMMENDER
# ============================================================

class RecommenderLSTM_ANN_Attention(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        fc_dim_1,
        fc_dim_2,
        dropout=0.3
    ):

        super().__init__()

        # ----------------------------------------------------
        # Bi-LSTM
        # ----------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
            bidirectional=True
        )

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        self.attention = Attention(
            hidden_dim
        )

        # ----------------------------------------------------
        # ANN
        # ----------------------------------------------------

        self.fc1 = nn.Linear(
            hidden_dim * 2,
            fc_dim_1
        )

        self.fc2 = nn.Linear(
            fc_dim_1,
            fc_dim_2
        )

        self.fc3 = nn.Linear(
            fc_dim_2,
            1
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        user_embedding,
        item_embedding
    ):

        # ----------------------------------------------------
        # Create a two-step sequence
        # Step 1 = user embedding
        # Step 2 = item embedding
        # ----------------------------------------------------

        sequence = torch.stack(
            [
                user_embedding,
                item_embedding
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Bi-LSTM
        # ----------------------------------------------------

        lstm_output, _ = self.lstm(
            sequence
        )

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        context_vector, attention_weights = (
            self.attention(lstm_output)
        )

        # ----------------------------------------------------
        # ANN
        # ----------------------------------------------------

        out = self.fc1(
            context_vector
        )

        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc2(out)

        out = self.relu(out)

        out = self.dropout(out)

        logits = self.fc3(out)

        return (
            logits,
            attention_weights
        )


# ============================================================
# 26. INITIALIZE MODEL
# ============================================================

model = RecommenderLSTM_ANN_Attention(
    input_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    fc_dim_1=FC_DIM_1,
    fc_dim_2=FC_DIM_2,
    dropout=DROPOUT
)

model = model.to(DEVICE)

print("\n================ MODEL ================")

print(model)


# ============================================================
# 27. LOSS / OPTIMIZER
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 28. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    dataloader,
    criterion,
    threshold=0.5
):

    model.eval()

    total_loss = 0.0

    all_probabilities = []
    all_labels = []

    all_attention = []

    with torch.no_grad():

        for (
            user_embeddings_batch,
            item_embeddings_batch,
            labels_batch
        ) in dataloader:

            user_embeddings_batch = (
                user_embeddings_batch.to(DEVICE)
            )

            item_embeddings_batch = (
                item_embeddings_batch.to(DEVICE)
            )

            labels_batch = (
                labels_batch
                .float()
                .unsqueeze(1)
                .to(DEVICE)
            )

            logits, attention_weights = model(
                user_embeddings_batch,
                item_embeddings_batch
            )

            loss = criterion(
                logits,
                labels_batch
            )

            total_loss += loss.item()

            probabilities = torch.sigmoid(
                logits
            )

            all_probabilities.extend(
                probabilities
                .cpu()
                .numpy()
                .flatten()
                .tolist()
            )

            all_labels.extend(
                labels_batch
                .cpu()
                .numpy()
                .flatten()
                .tolist()
            )

            all_attention.append(
                attention_weights
                .cpu()
                .numpy()
            )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    probabilities = np.array(
        all_probabilities
    )

    labels = np.array(
        all_labels
    )

    # --------------------------------------------------------
    # Binary predictions
    # --------------------------------------------------------

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    if len(np.unique(labels)) == 2:

        auc = roc_auc_score(
            labels,
            probabilities
        )

    else:

        auc = np.nan

    average_loss = (
        total_loss / len(dataloader)
    )

    metrics = {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

    return (
        metrics,
        probabilities,
        labels,
        predictions,
        all_attention
    )


# ============================================================
# 29. TRAINING LOOP
#     NO EARLY STOPPING
# ============================================================

training_history = []

print("\n================ TRAINING ================\n")

for epoch in range(NUM_EPOCHS):

    model.train()

    total_train_loss = 0.0

    for (
        user_embeddings_batch,
        item_embeddings_batch,
        labels_batch
    ) in train_loader:

        user_embeddings_batch = (
            user_embeddings_batch.to(DEVICE)
        )

        item_embeddings_batch = (
            item_embeddings_batch.to(DEVICE)
        )

        labels_batch = (
            labels_batch
            .float()
            .unsqueeze(1)
            .to(DEVICE)
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits, _ = model(
            user_embeddings_batch,
            item_embeddings_batch
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            labels_batch
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = (
        total_train_loss / len(train_loader)
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_metrics, _, _, _, _ = evaluate_model(
        model,
        val_loader,
        criterion,
        threshold=CLASSIFICATION_THRESHOLD
    )

    # --------------------------------------------------------
    # Store training history
    # --------------------------------------------------------

    training_history.append({
        "epoch": epoch + 1,
        "train_loss": average_train_loss,
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "val_auc": val_metrics["auc"]
    })

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(
        f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
        f"Train Loss: {average_train_loss:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Precision: {val_metrics['precision']:.4f} | "
        f"Val Recall: {val_metrics['recall']:.4f} | "
        f"Val F1: {val_metrics['f1']:.4f} | "
        f"Val AUC: {val_metrics['auc']:.4f}"
    )


# ============================================================
# 30. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    training_history
)

history_df.to_csv(
    "kexprs_training_history_movie.csv",
    index=False
)

print(
    "\nTraining history saved to "
    "kexprs_training_history_movie.csv"
)


# ============================================================
# 31. FINAL VALIDATION EVALUATION
# ============================================================

val_metrics, val_probabilities, val_labels, val_predictions, _ = (
    evaluate_model(
        model,
        val_loader,
        criterion,
        threshold=CLASSIFICATION_THRESHOLD
    )
)


print("\n================ FINAL VALIDATION ================")

print(
    f"Validation Loss:      {val_metrics['loss']:.4f}"
)

print(
    f"Validation Accuracy:  {val_metrics['accuracy']:.4f}"
)

print(
    f"Validation Precision: {val_metrics['precision']:.4f}"
)

print(
    f"Validation Recall:    {val_metrics['recall']:.4f}"
)

print(
    f"Validation F1:        {val_metrics['f1']:.4f}"
)

print(
    f"Validation AUC:       {val_metrics['auc']:.4f}"
)


# ============================================================
# 32. FINAL TEST EVALUATION
# ============================================================

test_metrics, test_probabilities, test_labels, test_predictions, test_attention = (
    evaluate_model(
        model,
        test_loader,
        criterion,
        threshold=CLASSIFICATION_THRESHOLD
    )
)


print("\n================ FINAL TEST RESULTS ================")

print(
    f"Test Loss:      {test_metrics['loss']:.4f}"
)

print(
    f"Test Accuracy:  {test_metrics['accuracy']:.4f}"
)

print(
    f"Test Precision: {test_metrics['precision']:.4f}"
)

print(
    f"Test Recall:    {test_metrics['recall']:.4f}"
)

print(
    f"Test F1:        {test_metrics['f1']:.4f}"
)

print(
    f"Test AUC:       {test_metrics['auc']:.4f}"
)


# ============================================================
# 33. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions
)

print("\n================ CONFUSION MATRIX ================")

print(cm)


# ============================================================
# 34. SAVE TEST PREDICTIONS
# ============================================================

test_results = recommendation_test.copy()

test_results["predicted_probability"] = (
    test_probabilities
)

test_results["predicted_label"] = (
    test_predictions
)

test_results.to_csv(
    "kexprs_test_predictions_movie.csv",
    index=False
)


# ============================================================
# 35. SAVE FINAL MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),

        "embedding_dim": EMBEDDING_DIM,

        "hidden_dim": HIDDEN_DIM,

        "num_layers": NUM_LAYERS,

        "fc_dim_1": FC_DIM_1,

        "fc_dim_2": FC_DIM_2,

        "dropout": DROPOUT,

        "classification_threshold": (
            CLASSIFICATION_THRESHOLD
        ),

        "num_epochs": NUM_EPOCHS,

        "learning_rate": LEARNING_RATE,

        "weight_decay": WEIGHT_DECAY,

        "test_metrics": test_metrics
    },

    "kexprs_recommender_movie.pt"
)


# ============================================================
# 36. SAVE FINAL METRICS
# ============================================================

metrics_df = pd.DataFrame([
    {
        "Dataset": "Validation",
        "Loss": val_metrics["loss"],
        "Accuracy": val_metrics["accuracy"],
        "Precision": val_metrics["precision"],
        "Recall": val_metrics["recall"],
        "F1": val_metrics["f1"],
        "AUC": val_metrics["auc"]
    },

    {
        "Dataset": "Test",
        "Loss": test_metrics["loss"],
        "Accuracy": test_metrics["accuracy"],
        "Precision": test_metrics["precision"],
        "Recall": test_metrics["recall"],
        "F1": test_metrics["f1"],
        "AUC": test_metrics["auc"]
    }
])

metrics_df.to_csv(
    "kexprs_movie_results.csv",
    index=False
)


# ============================================================
# 37. FINAL SUMMARY
# ============================================================

print("\n====================================================")
print("K-ExPReS MOVIELENS EXPERIMENT COMPLETED")
print("WITHOUT EARLY STOPPING")
print("====================================================")

print(
    "\nFinal Validation Metrics:"
)

print(
    f"Precision = {val_metrics['precision']:.4f}"
)

print(
    f"Recall    = {val_metrics['recall']:.4f}"
)

print(
    f"F1        = {val_metrics['f1']:.4f}"
)

print(
    f"AUC       = {val_metrics['auc']:.4f}"
)

print(
    "\nFinal Test Metrics:"
)

print(
    f"Precision = {test_metrics['precision']:.4f}"
)

print(
    f"Recall    = {test_metrics['recall']:.4f}"
)

print(
    f"F1        = {test_metrics['f1']:.4f}"
)

print(
    f"AUC       = {test_metrics['auc']:.4f}"
)

print("\nGenerated files:")

print("  recommendation_train.csv")
print("  recommendation_val.csv")
print("  recommendation_test.csv")
print("  kexprs_training_history_movie.csv")
print("  kexprs_test_predictions_movie.csv")
print("  kexprs_recommender_movie.pt")
print("  kexprs_movie_results.csv")

print("\n====================================================")