
# Fraud Detection with Django

## Overview

The Fraud Detection project aims to identify fraudulent Instagram profiles using machine learning techniques. Leveraging the power of Django, this web application provides users with a user-friendly interface to register and detect potentially fake accounts based on various criteria.

## Features

- **User Registration**: Users can register on the platform by providing relevant information about the Instagram profile they want to verify.
- **Machine Learning Model**: Utilizes a trained machine learning model to analyze input data and predict the likelihood of an Instagram profile being fake.
- **Real-time Detection**: Provides real-time feedback to users about the legitimacy of the submitted Instagram profile.
- **Responsive Interface**: Built with Bootstrap and designed to be responsive across different devices and screen sizes.

## Criteria for Fake Account Detection

When detecting potentially fake accounts, the application considers various criteria to assess their legitimacy. These criteria include:

1. **Profile Picture**: Fake accounts often use generic or stolen profile pictures. The absence of a profile picture or the use of a low-quality or inconsistent profile picture may indicate a fake account.

2. **Username**: Fake accounts may have usernames that are randomly generated or contain a string of numbers and letters. Unusual or nonsensical usernames could raise suspicion.

3. **Number of Followers**: Fake accounts typically have a disproportionate number of followers compared to the number of posts or engagement. An unusually high number of followers with very few posts or interactions could signal a fake account.

4. **Number of Follows**: Similarly, fake accounts may follow an unusually large number of users without engaging with their content. A significantly higher number of follows compared to followers could be a red flag.

5. **Number of Posts**: Fake accounts often have a low number of posts or may not have posted any content at all. A profile with a sparse posting history or an absence of recent activity might be suspicious.

6. **Description**: The profile description or bio of a fake account may contain generic phrases, misspellings, or nonsensical text. Lack of meaningful information or an overly promotional tone could indicate a fake account.

7. **External URLs**: Fake accounts may include links to suspicious websites or affiliate links in their profile description. Presence of irrelevant or questionable URLs could suggest a fake account.

8. **Privacy Settings**: Fake accounts may set their profiles to private to avoid scrutiny or detection. However, some fake accounts may also be set to public to appear more legitimate.

By analyzing these criteria and possibly more depending on the complexity of the detection algorithm, the application can determine the likelihood of an Instagram profile being fake. Machine learning models trained on datasets containing labeled examples of fake and genuine accounts can further enhance the accuracy of the detection process.

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/your_username/fraud-detection.git
    ```

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Run the Django server:
    ```
    python manage.py runserver
    ```

4. Access the application in your web browser at `http://localhost:8000`.

## Usage

1. Navigate to the homepage of the application.
2. Click on the "Register" button to submit information about the Instagram profile you want to verify.
3. Receive real-time feedback about the legitimacy of the submitted profile.

## Technologies Used

- **Python**: Programming language used for backend development and machine learning model training.
- **Django**: Web framework used for building the user interface and handling backend logic.
- **TensorFlow**: Deep learning library used for training the machine learning model.
- **Bootstrap**: Frontend framework used for designing the responsive user interface.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any inquiries or feedback, please contact:

- Suchir V
- suchirreddy31@gmail.com
