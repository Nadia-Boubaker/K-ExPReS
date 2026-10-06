from rdflib.plugins.sparql import prepareQuery

# Fonction pour obtenir les topics de connaissance
def get_knowledge_topics(g, course_uri):
    query = f"""
    SELECT ?topic WHERE {{
        <{course_uri}> <https://coursera.graph.edu/hasKnowledgeTopic> ?topic .
    }}
    """
    results = g.query(query)
    topics = [str(row[0]) for row in results]
    return topics

# print("Function to get knowledge topics defined.")

# Obtenir les topics de connaissance pour le cours recommandé
knowledge_topics = get_knowledge_topics(g, recommended_course_uri)

print("Knowledge topics for the recommended course:")
for topic in knowledge_topics:
    print(f"- {topic}")


def get_courses_with_same_topics_rated_by_user(g, knowledge_topics, user_uri, recommended_course_uri):
    courses = set()
    for topic in knowledge_topics:
        query = f"""
        SELECT ?course WHERE {{
            ?course <https://coursera.graph.edu/hasKnowledgeTopic> <{topic}> .
            <{user_uri}> ?interest ?course .
            FILTER (?interest IN (<https://coursera.graph.edu/HighInterest>, <https://coursera.graph.edu/MediumInterest>, <https://coursera.graph.edu/SmallInterest>))
            FILTER (?course != <{recommended_course_uri}>)
        }}
        """
        results = g.query(query)
        for row in results:
            courses.add(str(row[0]))
    return courses

# print("Function to get courses with same knowledge topics and rated by the user defined.")

user_uri = id_to_entity[user_id]

# Obtenir les cours ayant les mêmes topics de connaissance et évalués par le même utilisateur
courses_with_same_topics = get_courses_with_same_topics_rated_by_user(g, knowledge_topics, user_uri, recommended_course_uri)

print("Courses with the same knowledge topics and rated by the same user:")
for course in courses_with_same_topics:
    print(f"- {course}")

def get_course_details(g, course_uris):
    course_details = []

    for course_uri in course_uris:
        # print(f"Querying details for: {course_uri}")
        query = f"""
        SELECT ?courseName ?rating ?university ?level WHERE {{
            <{course_uri}> <correct_predicate_for_courseName> ?courseName .
            <{course_uri}> <correct_predicate_for_rating> ?rating .
            <{course_uri}> <correct_predicate_for_university> ?university .
            <{course_uri}> <correct_predicate_for_level> ?level .
        }}
        """
        results = g.query(query)
        for row in results:
            print("Found details: ", row)
            course_info = {
                "course_uri": course_uri,
                "course_name": str(row.courseName),
                "rating": str(row.rating),
                "university": str(row.university),
                "level": str(row.level)
            }
            course_details.append(course_info)

    return course_details

# Use the function to extract details of courses with the same knowledge topics and rated by the user
course_details = get_course_details(g, courses_with_same_topics)

# Extract and display information for each course
print("Detailed information about each course with the same knowledge topics and rated by the user:")
for course_uri in courses_with_same_topics:
    print(f"\nInformation for {course_uri}:")
    course_info = extract_information(g, course_uri)

    for predicate, objects in course_info.items():
        print(f"{predicate}:")
        for obj in objects:
            print(f"  - {obj}")



def find_paths_between_courses(g, start_course, end_courses, max_depth=3):
    paths = []
    for end_course in end_courses:
        if start_course == end_course:
            continue
        for depth in range(1, max_depth + 1):
            if depth == 1:
                query = f"""
                SELECT ?p1 ?intermediate WHERE {{
                    <{start_course}> ?p1 <{end_course}> .
                }}
                """
            else:
                query = f"""
                SELECT ?p1 ?intermediate ?p2 WHERE {{
                    <{start_course}> ?p1 ?intermediate .
                    ?intermediate ?p2 <{end_course}> .
                }}
                """
            results = g.query(query)
            for row in results:
                if depth == 1:
                    p1 = str(row[0])
                    paths.append((start_course, p1, end_course))
                else:
                    p1 = str(row[0])
                    intermediate = str(row[1])
                    p2 = str(row[2])
                    paths.append((start_course, p1, intermediate, p2, end_course))
    return paths

# print("Function to find paths between courses defined.")


# Trouver les chemins reliant le cours recommandé aux autres cours
paths = find_paths_between_courses(g, recommended_course_uri, courses_with_same_topics)

print("Paths between the recommended course and other courses with the same knowledge topics and rated by the same user:")
for path in paths:
    if len(path) == 3:
        print(f"Direct path from {path[0]} to {path[2]} via {path[1]}")
    elif len(path) == 5:
         print(f"Path from {path[0]} to {path[4]} via {path[1]} to {path[2]} and then via {path[3]}")

