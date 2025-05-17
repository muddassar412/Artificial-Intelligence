def suggest_career(scaled_data):
    """
    Suggests a suitable career based on scaled input data.

    Args:
        scaled_data (dict): A dictionary where keys represent skills/interests
                             and values are numerical scales (e.g., 1-5).

    Returns:
        tuple: A tuple containing the suggested career (str) and the rationale (str).
               Returns (None, "No suitable career found based on the input.") if no match.
    """
    # This is a very basic rule-based system. A real application would use
    # more sophisticated methods like machine learning.
    career_profiles = {
        "Software Engineer": {"technical_aptitude": 4, "problem_solving": 5, "creativity": 3},
        "Data Scientist": {"analytical_skills": 5, "mathematical_ability": 4, "communication": 3},
        "Marketing Manager": {"communication": 5, "creativity": 4, "analytical_skills": 3},
        "Financial Analyst": {"mathematical_ability": 5, "analytical_skills": 4, "attention_to_detail": 4},
        "UX Designer": {"creativity": 5, "empathy": 4, "technical_aptitude": 3},
        "Teacher": {"communication": 5, "patience": 5, "organization": 4},
    }

    best_match = None
    highest_score = -1
    rationale = ""

    for career, profile in career_profiles.items():
        score = 0
        match_count = 0
        for skill, required_level in profile.items():
            if skill in scaled_data:
                match_count += 1
                # A simple scoring based on how close the user's scale is to the requirement
                score += (6 - abs(scaled_data[skill] - required_level))

        if match_count > 0: # Only consider careers with some matching skills
            average_score = score / match_count
            if average_score > highest_score:
                highest_score = average_score
                best_match = career
                rationale = f"The suggested career '{career}' aligns well with your strengths in "
                rationale += ", ".join([skill for skill in profile if skill in scaled_data])
                rationale += f" with an average alignment score of {average_score:.2f}."

    return best_match, rationale

def get_provider_format(career):
    """
    Suggests a provider format for a given career.

    Args:
        career (str): The name of the career.

    Returns:
        str: A description of potential provider formats.
    """
    provider_formats = {
        "Software Engineer": "Technology companies (startups, large corporations), freelance platforms, open-source projects.",
        "Data Scientist": "Tech companies, research institutions, consulting firms, financial organizations, healthcare.",
        "Marketing Manager": "Companies across various industries, advertising agencies, marketing firms, non-profit organizations.",
        "Financial Analyst": "Investment banks, hedge funds, financial consulting firms, corporate finance departments.",
        "UX Designer": "Technology companies, design agencies, freelance, startups.",
        "Teacher": "Schools (public and private), universities, online education platforms, tutoring services.",
    }
    return provider_formats.get(career, "Information on provider formats for this career is not currently available.")

def main():
    """
    Main function to demonstrate the career counseling process.
    """
    print("Welcome to the AI Career Counselor!")
    print("Please rate the following skills/interests on a scale of 1 to 5 (1: Low, 5: High):")

    skills_interests = {
        "technical_aptitude": "Your interest in technology and how things work?",
        "problem_solving": "Your ability to analyze and solve complex issues?",
        "creativity": "Your inclination towards innovation and generating new ideas?",
        "analytical_skills": "Your ability to interpret data and draw conclusions?",
        "mathematical_ability": "Your comfort and proficiency with numbers and calculations?",
        "communication": "Your skill in expressing ideas clearly and effectively?",
        "attention_to_detail": "Your focus on accuracy and thoroughness?",
        "empathy": "Your ability to understand and share the feelings of others?",
        "patience": "Your capacity to remain calm and understanding in challenging situations?",
        "organization": "Your ability to structure and manage tasks effectively?",
    }

    scaled_data = {}
    for skill, question in skills_interests.items():
        while True:
            try:
                rating = int(input(f"{question} (1-5): "))
                if 1 <= rating <= 5:
                    scaled_data[skill] = rating
                    break
                else:
                    print("Rating must be between 1 and 5. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 5.")

    suggested_career, rationale = suggest_career(scaled_data)

    if suggested_career:
        print("\nBased on your input:")
        print(f"Suggested Career: {suggested_career}")
        print(f"Rationale: {rationale}")
        provider_format = get_provider_format(suggested_career)
        print(f"Potential Provider Formats: {provider_format}")
    else:
        print("\nNo suitable career could be suggested based on your input.")
if __name__ == "__main__":
    main()
