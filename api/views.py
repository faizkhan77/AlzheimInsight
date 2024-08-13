from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
import numpy as np
import joblib
import pandas as pd
from django.views.decorators.csrf import csrf_exempt


# Load the trained model
model = joblib.load("artifacts/catboost_alzheimer_model.pickle")

# Loading saved Pipeline which i used to transform the non-binary columns
scaler = joblib.load("artifacts/non_binary_transformer.pkl")


# Global Features variables
__Age = None
__Ethnicity = None
__EducationLevel = None
__BMI = None
__AlcoholConsumption = None
__PhysicalActivity = None
__DietQuality = None
__SleepQuality = None
__SystolicBP = None
__DiastolicBP = None
__CholesterolTotal = None
__CholesterolLDL = None
__CholesterolHDL = None
__CholesterolTriglycerides = None
__MMSE = None
__FunctionalAssessment = None
__ADL = None
__RiskFactorScore = None
__SymptomCount = None
__gender = None
__smoking = None
__family_history = None
__cvd = None
__diabetes = None
__depression = None
__head_injury = None
__hypertension = None
__memory_complaints = None
__behavioral_problems = None
__confusion = None
__disorientation = None
__personality_changes = None
__difficulty_tasks = None
__forgetfulnes = None
__diabetes_cv = None
__has_any_risk_factor = None
__has_any_symptom = None
__age_group_70_80 = None
__age_group_80_90 = None
__age_group_90_plus = None


def scaling():
    global __Age
    global __Ethnicity
    global __EducationLevel
    global __BMI
    global __AlcoholConsumption
    global __PhysicalActivity
    global __DietQuality
    global __SleepQuality
    global __SystolicBP
    global __DiastolicBP
    global __CholesterolTotal
    global __CholesterolLDL
    global __CholesterolHDL
    global __CholesterolTriglycerides
    global __MMSE
    global __FunctionalAssessment
    global __ADL
    global __RiskFactorScore
    global __SymptomCount
    global __gender
    global __smoking
    global __family_history
    global __cvd
    global __diabetes
    global __depression
    global __head_injury
    global __hypertension
    global __memory_complaints
    global __behavioral_problems
    global __confusion
    global __disorientation
    global __personality_changes
    global __difficulty_tasks
    global __forgetfulnes
    global __diabetes_cv
    global __has_any_risk_factor
    global __has_any_symptom
    global __age_group_70_80
    global __age_group_80_90
    global __age_group_90_plus

    # Columns names for Non binary features only since thats what need to be scaled
    column_names = [
        "Age",
        "Ethnicity",
        "EducationLevel",
        "BMI",
        "AlcoholConsumption",
        "PhysicalActivity",
        "DietQuality",
        "SleepQuality",
        "SystolicBP",
        "DiastolicBP",
        "CholesterolTotal",
        "CholesterolLDL",
        "CholesterolHDL",
        "CholesterolTriglycerides",
        "MMSE",
        "FunctionalAssessment",
        "ADL",
        "RiskFactorScore",
        "SymptomCount",
    ]

    # Creaing DataFrame (because our Scaler understand the DataFrame structure only)
    features_df = pd.DataFrame(
        [
            [
                __Age,
                __Ethnicity,
                __EducationLevel,
                __BMI,
                __AlcoholConsumption,
                __PhysicalActivity,
                __DietQuality,
                __SleepQuality,
                __SystolicBP,
                __DiastolicBP,
                __CholesterolTotal,
                __CholesterolLDL,
                __CholesterolHDL,
                __CholesterolTriglycerides,
                __MMSE,
                __FunctionalAssessment,
                __ADL,
                __RiskFactorScore,
                __SymptomCount,
            ]
        ],
        columns=column_names,
    )

    # Transforming the DataFrame datas
    scaled_features = scaler.transform(features_df)

    # Unpack the scaled datas back into their respective variables
    (
        __Age,
        __Ethnicity,
        __EducationLevel,
        __BMI,
        __AlcoholConsumption,
        __PhysicalActivity,
        __DietQuality,
        __SleepQuality,
        __SystolicBP,
        __DiastolicBP,
        __CholesterolTotal,
        __CholesterolLDL,
        __CholesterolHDL,
        __CholesterolTriglycerides,
        __MMSE,
        __FunctionalAssessment,
        __ADL,
        __RiskFactorScore,
        __SymptomCount,
    ) = scaled_features[0]

    return scaled_features


def prediction():
    predicted = model.predict(
        [
            __Age,
            __Ethnicity,
            __EducationLevel,
            __BMI,
            __AlcoholConsumption,
            __PhysicalActivity,
            __DietQuality,
            __SleepQuality,
            __SystolicBP,
            __DiastolicBP,
            __CholesterolTotal,
            __CholesterolLDL,
            __CholesterolHDL,
            __CholesterolTriglycerides,
            __MMSE,
            __FunctionalAssessment,
            __ADL,
            __RiskFactorScore,
            __SymptomCount,
            __gender,
            __smoking,
            __family_history,
            __cvd,
            __diabetes,
            __depression,
            __head_injury,
            __hypertension,
            __memory_complaints,
            __behavioral_problems,
            __confusion,
            __disorientation,
            __personality_changes,
            __difficulty_tasks,
            __forgetfulnes,
            __diabetes_cv,
            __has_any_risk_factor,
            __has_any_symptom,
            __age_group_70_80,
            __age_group_80_90,
            __age_group_90_plus,
        ]
    )

    return predicted


@csrf_exempt
def predict_alzheimer(request):

    global __Age
    global __Ethnicity
    global __EducationLevel
    global __BMI
    global __AlcoholConsumption
    global __PhysicalActivity
    global __DietQuality
    global __SleepQuality
    global __SystolicBP
    global __DiastolicBP
    global __CholesterolTotal
    global __CholesterolLDL
    global __CholesterolHDL
    global __CholesterolTriglycerides
    global __MMSE
    global __FunctionalAssessment
    global __ADL
    global __RiskFactorScore
    global __SymptomCount
    global __gender
    global __smoking
    global __family_history
    global __cvd
    global __diabetes
    global __depression
    global __head_injury
    global __hypertension
    global __memory_complaints
    global __behavioral_problems
    global __confusion
    global __disorientation
    global __personality_changes
    global __difficulty_tasks
    global __forgetfulnes
    global __diabetes_cv
    global __has_any_risk_factor
    global __has_any_symptom
    global __age_group_70_80
    global __age_group_80_90
    global __age_group_90_plus

    # Getting all the inputs from the Form in `modelform.html`
    if request.method == "POST":
        __Age = int(request.POST.get("age"))
        __Ethnicity = (
            0
            if request.POST.get("ethnicity") == "caucasian"
            else (
                1
                if request.POST.get("ethnicity") == "african_american"
                else 2 if request.POST.get("ethnicity") == "asian" else 3
            )
        )
        __EducationLevel = (
            0
            if request.POST.get("education") == "none"
            else (
                1
                if request.POST.get("education") == "high_school"
                else 2 if request.POST.get("education") == "bachelor" else 3
            )
        )
        __BMI = float(request.POST.get("bmi"))
        __AlcoholConsumption = float(request.POST.get("alcohol"))
        __PhysicalActivity = float(request.POST.get("physical_activity"))
        __DietQuality = float(request.POST.get("diet_quality"))
        __SleepQuality = float(request.POST.get("sleep_quality"))

        __SystolicBP = float(request.POST.get("SystolicBP"))
        __DiastolicBP = float(request.POST.get("DiastolicBP"))
        __CholesterolTotal = float(request.POST.get("CholesterolTotal"))
        __CholesterolLDL = float(request.POST.get("CholesterolLDL"))
        __CholesterolHDL = float(request.POST.get("CholesterolHDL"))
        __CholesterolTriglycerides = float(request.POST.get("CholesterolTriglycerides"))

        __MMSE = float(request.POST.get("mmse"))
        __FunctionalAssessment = float(request.POST.get("functional_assessment"))
        __ADL = float(request.POST.get("adl"))

        __gender = 1 if request.POST.get("gender") == "male" else 0
        __smoking = 1 if request.POST.get("smoking") == "yes" else 0

        __family_history = (
            1 if request.POST.get("family_history_alzheimers") == "yes" else 0
        )
        __cvd = 1 if request.POST.get("cardiovascular_disease") == "yes" else 0
        __diabetes = 1 if request.POST.get("diabetes") == "yes" else 0
        __depression = 1 if request.POST.get("depression") == "yes" else 0
        __head_injury = 1 if request.POST.get("head_injury") == "yes" else 0
        __hypertension = 1 if request.POST.get("hypertension") == "yes" else 0
        __memory_complaints = 1 if request.POST.get("memory_complaints") == "yes" else 0
        __behavioral_problems = (
            1 if request.POST.get("behavioral_problems") == "yes" else 0
        )

        __confusion = 1 if request.POST.get("Confusion") == "yes" else 0
        __disorientation = 1 if request.POST.get("Disorientation") == "yes" else 0
        __personality_changes = (
            1 if request.POST.get("PersonalityChanges") == "yes" else 0
        )
        __difficulty_tasks = (
            1 if request.POST.get("DifficultyCompletingTasks") == "yes" else 0
        )
        __forgetfulness = 1 if request.POST.get("Forgetfulness") == "yes" else 0

        __diabetes_cv = __diabetes * __cvd

        # -------------- Calculate derived features ---------------
        __RiskFactorScore = (
            __family_history
            + __cvd
            + __diabetes
            + __depression
            + __head_injury
            + __hypertension
        )
        __SymptomCount = (
            __confusion
            + __disorientation
            + __personality_changes
            + __difficulty_tasks
            + __forgetfulness
        )

        __has_any_risk_factor = 1 if __RiskFactorScore > 0 else 0
        __has_any_symptom = 1 if __SymptomCount > 0 else 0

        # One-hot encoding for age group
        __age_group_70_80 = 1 if 70 <= __Age < 80 else 0
        __age_group_80_90 = 1 if 80 <= __Age < 90 else 0
        __age_group_90_plus = 1 if __Age >= 90 else 0

        # ---------------------------------------------------------

        # Calling `scaling()` to scale and then `prediction()` to give predicted output
        scaled_features = scaling()
        predicted_output = prediction()

    var = {
        "predicted": predicted_output,
    }

    return render(request, "diagnose_result.html", var)
