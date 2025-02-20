import pandas as pd

import lotus
from lotus.models import LM, LiteLLMRM
from lotus.types import CascadeArgs, ProxyModel
from lotus.vector_store import FaissVS

gpt_4o_mini = LM("gpt-4o-mini")
gpt_4o = LM("gpt-4o")
rm = LiteLLMRM(model="text-embedding-3-small")
vs = FaissVS()

lotus.settings.configure(lm=gpt_4o, helper_lm=gpt_4o_mini, rm=rm, vs=vs)
data = {
    "Course Name": [
        "Probability and Random Processes",
        "Optimization Methods in Engineering",
        "Digital Design and Integrated Circuits",
        "Computer Security",
        "Data Structures and Algorithms",
        "Machine Learning",
        "Artificial Intelligence",
        "Natural Language Processing",
        "Introduction to Robotics",
        "Control Systems",
        "Linear Algebra and Differential Equations",
        "Database Systems",
        "Cloud Computing",
        "Software Engineering",
        "Operating Systems",
        "Discrete Mathematics",
        "Numerical Methods",
        "Wireless Communication Systems",
        "Embedded Systems",
        "Advanced Computer Architecture",
        "Graph Theory",
        "Cryptography and Network Security",
        "Big Data Analytics",
        "Deep Learning",
        "Organic Chemistry",
        "Molecular Biology",
        "Environmental Science",
        "Genetics and Evolution",
        "Human Physiology",
        "Introduction to Anthropology",
        "Cultural Studies",
        "Political Theory",
        "Macroeconomics",
        "Microeconomics",
        "Introduction to Sociology",
        "Developmental Psychology",
        "Cognitive Science",
        "Introduction to Philosophy",
        "Ethics and Moral Philosophy",
        "History of Western Civilization",
        "Art History: Renaissance to Modern",
        "World Literature",
        "Introduction to Journalism",
        "Public Speaking and Communication",
        "Creative Writing",
        "Music Theory",
        "Introduction to Theater",
        "Film Studies",
        "Environmental Policy and Law",
        "Sustainability and Renewable Energy",
        "Urban Planning and Design",
        "International Relations",
        "Marketing Principles",
        "Organizational Behavior",
        "Financial Accounting",
        "Corporate Finance",
        "Business Law",
        "Supply Chain Management",
        "Operations Research",
        "Entrepreneurship and Innovation",
        "Introduction to Psychology",
        "Health Economics",
        "Biostatistics",
        "Social Work Practice",
        "Public Health Policy",
        "Environmental Ethics",
        "History of Political Thought",
        "Quantitative Research Methods",
        "Comparative Politics",
        "Urban Economics",
        "Behavioral Economics",
        "Sociology of Education",
        "Social Psychology",
        "Gender Studies",
        "Media and Communication Studies",
        "Advertising and Brand Strategy",
        "Sports Management",
        "Introduction to Archaeology",
        "Ecology and Conservation Biology",
        "Marine Biology",
        "Geology and Earth Science",
        "Astronomy and Astrophysics",
        "Introduction to Meteorology",
        "Introduction to Oceanography",
        "Quantum Physics",
        "Thermodynamics",
        "Fluid Mechanics",
        "Solid State Physics",
        "Classical Mechanics",
        "Introduction to Civil Engineering",
        "Material Science and Engineering",
        "Structural Engineering",
        "Environmental Engineering",
        "Energy Systems Engineering",
        "Aerodynamics",
        "Heat Transfer",
        "Renewable Energy Systems",
        "Transportation Engineering",
        "Water Resources Management",
        "Principles of Accounting",
        "Project Management",
        "International Business",
        "Business Analytics",
    ]
}
df = pd.DataFrame(data)
user_instruction = "{Course Name} requires a lot of math"

# We can cascade with a helper LM
cascade_args = CascadeArgs(
    recall_target=0.9,
    precision_target=0.9,
    sampling_percentage=0.5,
    failure_probability=0.2,
    proxy_model=ProxyModel.HELPER_LM,
)

filtered_df, stats = df.sem_filter(user_instruction=user_instruction, cascade_args=cascade_args, return_stats=True)
print(filtered_df)
print(stats)

# Or we can cascade with an embedding model (just set the proxy model to EMBEDDING_MODEL)
df = df.sem_index("Course Name", "index_dir")
cascade_args = CascadeArgs(
    recall_target=0.9,
    precision_target=0.9,
    sampling_percentage=0.5,
    failure_probability=0.2,
    proxy_model=ProxyModel.EMBEDDING_MODEL,
)
filtered_df, stats = df.sem_filter(user_instruction=user_instruction, cascade_args=cascade_args, return_stats=True)
print(filtered_df)
print(stats)
