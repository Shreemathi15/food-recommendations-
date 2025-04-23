import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


# Function to calculate daily caloric and macronutrient requirements
def calculate_nutrition(height: float, weight: float, age: int, gender: str, activity_level: str) -> tuple:
    if gender.lower() == "male":
        bmr = 88.36 + (13.4 * weight) + (4.8 * height) - (5.7 * age)
    else:
        bmr = 447.6 + (9.2 * weight) + (3.1 * height) - (4.3 * age)

    activity_multiplier = {
        "sedentary": 1.2,
        "lightly active": 1.5,
        "moderately active": 1.8,
        "very active": 2.2,
    }

    daily_calories = bmr * activity_multiplier.get(activity_level.lower(), 1.2)
    meal_calories = daily_calories / 3

    protein_grams = weight * (1.2 if activity_level == "sedentary" else 2.0)
    fat_grams = (0.3 * daily_calories) / 9
    carbs_grams = (daily_calories - (protein_grams * 4 + fat_grams * 9)) / 4

    meal_protein = protein_grams / 3
    meal_fat = fat_grams / 3
    meal_carbs = carbs_grams / 3

    return max(300, min(400, meal_calories)), meal_protein, meal_fat, meal_carbs


def scaling(dataframe):
    scaler = StandardScaler()
    prep_data = scaler.fit_transform(dataframe.iloc[:, 6:10].to_numpy())
    return prep_data, scaler


def nn_predictor(prep_data):
    neigh = NearestNeighbors(metric='cosine', algorithm='brute')
    neigh.fit(prep_data)
    return neigh


def build_pipeline(neigh, scaler, params):
    transformer = FunctionTransformer(neigh.kneighbors, kw_args=params)
    pipeline = Pipeline([('std_scaler', scaler), ('NN', transformer)])
    return pipeline


def extract_data(dataframe, ingredients):
    extracted_data = dataframe.copy()
    extracted_data = extract_ingredient_filtered_data(extracted_data, ingredients)
    return extracted_data


def extract_ingredient_filtered_data(dataframe, ingredients):
    if not ingredients:
        return dataframe

    regex_pattern = r'\b(?:' + '|'.join(map(re.escape, ingredients)) + r')\b'
    filtered_dataframe = dataframe[
        ~dataframe['RecipeIngredientParts'].str.contains(regex_pattern, regex=True, flags=re.IGNORECASE, na=False)
    ]
    return filtered_dataframe


def apply_pipeline(pipeline, _input, extracted_data):
    _input = np.array(_input).reshape(1, -1)
    return extracted_data.iloc[pipeline.transform(_input)[0]]


def recommend(dataframe, _input, ingredients=[], params={'n_neighbors': 5, 'return_distance': False}):
    extracted_data = extract_data(dataframe, ingredients)
    if extracted_data.shape[0] >= params['n_neighbors']:
        prep_data, scaler = scaling(extracted_data)
        neigh = nn_predictor(prep_data)
        pipeline = build_pipeline(neigh, scaler, params)
        return apply_pipeline(pipeline, _input, extracted_data)
    else:
        return None


def extract_quoted_strings(s):
    return re.findall(r'"([^"]*)"', s)


def output_recommended_recipes(dataframe):
    if dataframe is not None:
        output = dataframe.copy()
        output = output.to_dict("records")
        for recipe in output:
            recipe['RecipeIngredientParts'] = extract_quoted_strings(recipe['RecipeIngredientParts'])
            recipe['RecipeInstructions'] = extract_quoted_strings(recipe['RecipeInstructions'])
    else:
        output = None
    return output


def calculate_metrics(recommended_recipes, user_feedback):
    # Count correct and total recommendations
    correct = sum(1 for recipe, liked in zip(recommended_recipes, user_feedback) if liked)
    total = len(recommended_recipes)
    
    # Calculate metrics
    accuracy = correct / total if total else 0
    precision = correct / total if total else 0  # Precision equals accuracy in this scenario
    return accuracy, precision


def recommend_based_on_user_input(dataset, user_data, params={'n_neighbors': 5, 'return_distance': False}):
    calories_needed, protein_needed, fat_needed, carbs_needed = calculate_nutrition(
        user_data['height'], user_data['weight'],
        user_data['age'], user_data['gender'],
        user_data['activity_level']
    )

    filtered_recipes = extract_ingredient_filtered_data(dataset, user_data['allergies'])
    nutrition_input = [calories_needed, protein_needed, fat_needed, carbs_needed]
    recommendations = recommend(filtered_recipes, nutrition_input, ingredients=user_data['allergies'], params=params)
    
    # Example user feedback (1 for liked, 0 for not liked)
    user_feedback = [1, 0, 1, 1, 0]
    recommended_recipes = output_recommended_recipes(recommendations)

    if recommended_recipes:
        accuracy, precision = calculate_metrics(recommended_recipes, user_feedback)
        print(f"Accuracy: {accuracy * 100:.2f}%, Precision: {precision * 100:.2f}%")
    
    return recommended_recipes


# Example usage:
user_data = {
    'height': 170,
    'weight': 65,
    'age': 30,
    'gender': 'female',
    'activity_level': 'moderately active',
    'allergies': ['peanut', 'dairy'],
    'dietary_preference': 'vegetarian'
}