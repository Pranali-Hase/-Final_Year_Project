# ============================
# ML MODEL UTILITIES (UPDATED)
# ============================

import joblib


# ============================
# LOAD TRAINED MODEL
# ============================

model = joblib.load("model.pkl")


# ============================
# FEATURE EXTRACTION
# ============================

def extract_features(user, ingredients_text):

    text = ingredients_text.lower()

    allergies = user.get("allergies", "").lower()
    conditions = user.get("conditions", "").lower()
    diet = user.get("diet", "").lower()


    # ----------------------------
    # allergy_conflict
    # ----------------------------

    allergy_conflict = int(any(

        allergy.strip() in text

        for allergy in allergies.split(",")

        if allergy.strip()

    ))


    # ----------------------------
    # condition_conflict
    # ----------------------------

    condition_conflict = 0

    if "diabetes" in conditions and "sugar" in text:

        condition_conflict = 1

    elif ("bp" in conditions or "blood pressure" in conditions) and "salt" in text:

        condition_conflict = 1

    elif "heart" in conditions and "oil" in text:

        condition_conflict = 1


    # ----------------------------
    # diet_conflict
    # ----------------------------

    diet_conflict = 0

    if diet == "vegan" and any(x in text for x in ["milk","cheese","butter"]):

        diet_conflict = 1

    elif diet == "vegetarian" and any(x in text for x in ["chicken","fish","meat","egg"]):

        diet_conflict = 1


    return [[

        allergy_conflict,
        condition_conflict,
        diet_conflict

    ]]


# ============================
# PREDICT FUNCTION
# ============================

def predict_food(user, ingredients_text):

    features = extract_features(user, ingredients_text)

    prediction = model.predict(features)[0]


    return prediction


# ============================
# INGREDIENT RISK ANALYSIS
# ============================

def analyze_ingredient_risk(ingredients_text):

    text = ingredients_text.lower()


    HIGH = [

        "sugar",
        "trans fat",
        "glucose",
        "fructose"

    ]


    MEDIUM = [

        "salt",
        "milk",
        "oil",
        "preservative"

    ]


    LOW = [

        "rice",
        "wheat",
        "corn",
        "dal",
        "grain"

    ]


    high = [i for i in HIGH if i in text]

    medium = [i for i in MEDIUM if i in text]

    low = [i for i in LOW if i in text]


    return {

        "high": high,
        "medium": medium,
        "low": low

    }