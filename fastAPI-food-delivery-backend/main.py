from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
from model import recommend, output_recommended_recipes, calculate_nutrition
import os
from fastapi.middleware.cors import CORSMiddleware
import logging

# Instantiate FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset
dataset_path = os.path.join(os.getcwd(), "updated_recipes.csv")
print(f"Loading dataset from: {dataset_path}")
dataset = pd.read_csv(dataset_path)

# Debugging dataset loading
print("DietaryPreference" in dataset.columns)
print(dataset.columns)
print(dataset.head)

# Classes
class Params(BaseModel):
    n_neighbors: int = 5
    return_distance: bool = False


class DemographicInput(BaseModel):
    height: float
    weight: float
    age: int
    gender: str
    activity_level: str
    allergies: List[str] = []
    dietary_preference: str = "non-vegetarian"


class PredictionOut(BaseModel):
    output: Optional[List[dict]] = None


# Function to get ingredients to avoid based on dietary preferences
def get_avoid_ingredients(dietary_preference: str) -> List[str]:
    dietary_restrictions_map = {
        "vegetarian": ["meat", "chicken", "beef", "pork", "fish", "seafood"],
        "vegan": ["meat", "chicken", "beef", "pork", "fish", "seafood", "egg", "milk", "cheese", "butter", "honey"],
        "non-vegetarian": [],  # No restrictions
    }
    return dietary_restrictions_map.get(dietary_preference.lower(), [])


@app.post("/predict-demographic/", response_model=PredictionOut)
def predict_demographic(demographic_input: DemographicInput):
    logging.info(f"Received input: {demographic_input}")
    
    # Calculate caloric and macronutrient needs
    calories, protein, fat, carbs = calculate_nutrition(
        demographic_input.height,
        demographic_input.weight,
        demographic_input.age,
        demographic_input.gender,
        demographic_input.activity_level,
    )

    # Prepare model input with nutritional features
    nutrition_input = [calories, protein, fat, carbs]

    # Combine allergies and dietary restrictions into ingredients to avoid
    ingredients_to_avoid = demographic_input.allergies + get_avoid_ingredients(demographic_input.dietary_preference)

    # Filter dataset to exclude recipes with avoid ingredients in both columns
    def recipe_is_suitable(row):
        """
        Checks if a recipe is suitable by ensuring that no avoid ingredients
        are present in both RecipeIngredientParts and RecipeInstructions.
        """
        combined_text = f"{row.get('RecipeIngredientParts', '')} {row.get('RecipeInstructions', '')}"
        return not any(ingredient.lower() in combined_text.lower() for ingredient in ingredients_to_avoid)

    filtered_dataset = dataset[dataset.apply(recipe_is_suitable, axis=1)]

    # Get recommendations based on the filtered dataset
    recommendation_dataframe = recommend(filtered_dataset, nutrition_input, ingredients_to_avoid)

    # Debugging: log the output of recommendations
    if recommendation_dataframe is not None:
        print(f"Recommendations: {recommendation_dataframe.shape[0]} recommendations found")
    else:
        print("No recommendations found.")

    # Convert recommendations to output format
    output = output_recommended_recipes(recommendation_dataframe)

    if output:
        print("Recommended dishes:")
        for idx, dish in enumerate(output):
            print(f"{idx + 1}. {dish['Name']}")
    else:
        print("No recommendations available.")

    return {"output": output}
