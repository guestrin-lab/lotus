import pandas as pd
import time

import lotus
from lotus.models import LM

def main():
    lm = LM(model="gpt-4o-mini")

    lotus.settings.configure(lm=lm)
    lotus.settings.configure(enable_multithreading=True)

    # turn on lotus debug logging
    lotus.logger.setLevel("DEBUG")

    data = {
        "Course Name": [
            "Probability and Random Processes", "Optimization Methods in Engineering",
            "Digital Design and Integrated Circuits", "Computer Security", "Cooking",
            "Food Sciences", "Machine Learning", "Data Structures and Algorithms",
            "Quantum Mechanics", "Organic Chemistry", "Artificial Intelligence", "Robotics",
            "Thermodynamics", "Fluid Mechanics", "Molecular Biology", "Genetics",
            "Astrophysics", "Neuroscience", "Microeconomics", "Macroeconomics",
            "Linear Algebra", "Calculus", "Statistics", "Differential Equations",
            "Discrete Mathematics", "Number Theory", "Graph Theory", "Topology",
            "Complex Analysis", "Real Analysis", "Abstract Algebra", "Numerical Methods",
            "Cryptography", "Network Security", "Operating Systems", "Databases",
            "Computer Networks", "Software Engineering", "Compilers", "Computer Architecture",
            "Parallel Computing", "Distributed Systems", "Cloud Computing", "Big Data Analytics",
            "Natural Language Processing", "Computer Vision", "Reinforcement Learning",
            "Deep Learning", "Bioinformatics", "Computational Biology", "Systems Biology",
            "Biochemistry", "Physical Chemistry", "Inorganic Chemistry", "Analytical Chemistry",
            "Environmental Chemistry", "Materials Science", "Nanotechnology", "Optics",
            "Electromagnetism", "Nuclear Physics", "Particle Physics", "Cosmology",
            "Planetary Science", "Geophysics", "Atmospheric Science", "Oceanography",
            "Ecology", "Evolutionary Biology", "Botany", "Zoology", "Microbiology",
            "Immunology", "Virology", "Pharmacology", "Physiology", "Anatomy",
            "Neurobiology", "Cognitive Science", "Psychology", "Sociology", "Anthropology",
            "Archaeology", "Linguistics", "Philosophy", "Ethics", "Logic",
            "Political Science", "International Relations", "Public Policy", "Economics",
            "Finance", "Accounting", "Marketing", "Management", "Entrepreneurship",
            "Law", "Criminal Justice", "Human Rights", "Environmental Studies",
            "Sustainability", "Urban Planning", "Architecture", "Civil Engineering",
            "Mechanical Engineering", "Electrical Engineering", "Chemical Engineering",
            "Aerospace Engineering", "Biomedical Engineering", "Environmental Engineering"
        ],
        "Grade Level": [
            "High School", "Graduate", "Graduate", "High School", "Undergraduate",
            "Undergraduate", "High School", "Undergraduate", "High School", "Undergraduate",
            "High School", "Graduate", "Undergraduate", "Undergraduate", "Graduate",
            "Undergraduate", "Graduate", "Graduate", "Undergraduate", "Undergraduate",
            "Undergraduate", "Undergraduate", "High School", "High School", "Undergraduate",
            "Graduate", "Graduate", "Graduate", "High School", "Graduate", "Graduate", "Graduate",
            "Graduate", "High School", "Undergraduate", "High School", "Undergraduate",
            "Undergraduate", "Graduate", "Undergraduate", "Undergraduate", "Graduate", "Graduate",
            "Graduate", "Graduate", "Graduate", "Graduate", "Graduate", "Graduate", "Graduate",
            "Undergraduate", "Graduate", "Undergraduate", "High School", "Graduate", "Graduate",
            "Graduate", "High School", "Graduate", "High School", "Graduate", "Graduate",
            "Graduate", "Graduate", "Graduate", "Graduate", "Graduate", "Graduate",
            "High School", "High School", "High School", "Undergraduate", "Graduate",
            "Graduate", "Graduate", "High School", "Undergraduate", "Undergraduate",
            "Graduate", "Graduate", "Undergraduate", "Undergraduate", "Undergraduate",
            "High School", "High School", "Graduate", "Graduate", "High School", "Graduate",
            "Graduate", "Graduate", "Undergraduate", "Undergraduate", "Undergraduate", "Undergraduate",
            "High School", "High School", "Graduate", "Undergraduate", "Undergraduate", "Undergraduate",
            "Undergraduate", "Undergraduate", "Undergraduate", "Graduate", "Graduate",
            "Graduate", "Graduate", "Graduate", "Graduate"
        ],
    }

    df = pd.DataFrame(data)
    start_time = time.time()
    df = df.sem_agg("Summarize all {Course Name}", group_by=["Grade Level"])
    end_time = time.time()
    print(df._output[0])
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()