columns = [
    "disease", "anxiety_and_nervousness", "depression", "shortness_of_breath",
    "depressive_or_psychotic_symptoms", "sharp_chest_pain", "dizziness", "insomnia",
    "abnormal_involuntary_movements", "chest_tightness", "palpitations", "irregular_heartbeat",
    "breathing_fast", "hoarse_voice", "sore_throat", "difficulty_speaking", "cough",
    "nasal_congestion", "throat_swelling", "diminished_hearing", "lump_in_throat",
    "throat_feels_tight", "difficulty_in_swallowing", "skin_swelling", "retention_of_urine",
    "groin_mass", "leg_pain", "hip_pain", "suprapubic_pain", "blood_in_stool",
    "lack_of_growth", "emotional_symptoms", "elbow_weakness", "back_weakness",
    "pus_in_sputum", "symptoms_of_the_scrotum_and_testes", "swelling_of_scrotum",
    "pain_in_testicles", "flatulence", "pus_draining_from_ear", "jaundice",
    "mass_in_scrotum", "white_discharge_from_eye", "irritable_infant", "abusing_alcohol",
    "fainting", "hostile_behavior", "drug_abuse", "sharp_abdominal_pain", "feeling_ill",
    "vomiting", "headache", "nausea", "diarrhea", "vaginal_itching", "vaginal_dryness",
    "painful_urination", "involuntary_urination", "pain_during_intercourse", "frequent_urination",
    "lower_abdominal_pain", "vaginal_discharge", "blood_in_urine", "hot_flashes",
    "intermenstrual_bleeding", "hand_or_finger_pain", "wrist_pain", "hand_or_finger_swelling",
    "arm_pain", "wrist_swelling", "arm_stiffness_or_tightness", "arm_swelling",
    "hand_or_finger_stiffness_or_tightness", "wrist_stiffness_or_tightness", "lip_swelling",
    "toothache", "abnormal_appearing_skin", "skin_lesion", "acne_or_pimples", "dry_lips",
    "facial_pain", "mouth_ulcer", "skin_growth", "eye_deviation", "diminished_vision",
    "double_vision", "cross_eyed", "symptoms_of_eye", "pain_in_eye", "eye_moves_abnormally",
    "abnormal_movement_of_eyelid", "foreign_body_sensation_in_eye", "irregular_appearing_scalp",
    "swollen_lymph_nodes", "back_pain", "neck_pain", "low_back_pain", "pain_of_the_anus",
    "pain_during_pregnancy", "pelvic_pain", "impotence", "infant_spitting_up",
    "vomiting_blood", "regurgitation", "burning_abdominal_pain", "restlessness",
    "symptoms_of_infants", "wheezing", "peripheral_edema", "neck_mass", "ear_pain",
    "jaw_swelling", "mouth_dryness", "neck_swelling", "knee_pain", "foot_or_toe_pain",
    "bowlegged_or_knock_kneed", "ankle_pain", "bones_are_painful", "knee_weakness",
    "elbow_pain", "knee_swelling", "skin_moles", "knee_lump_or_mass", "weight_gain",
    "problems_with_movement", "knee_stiffness_or_tightness", "leg_swelling",
    "foot_or_toe_swelling", "heartburn", "smoking_problems", "muscle_pain",
    "infant_feeding_problem", "recent_weight_loss", "problems_with_shape_or_size_of_breast",
    "underweight", "difficulty_eating", "scanty_menstrual_flow", "vaginal_pain",
    "vaginal_redness", "vulvar_irritation", "weakness", "decreased_heart_rate",
    "increased_heart_rate", "bleeding_or_discharge_from_nipple", "ringing_in_ear",
    "plugged_feeling_in_ear", "itchy_ear", "frontal_headache", "fluid_in_ear",
    "neck_stiffness_or_tightness", "spots_or_clouds_in_vision", "eye_redness",
    "lacrimation", "itchiness_of_eye", "blindness", "eye_burns_or_stings",
    "itchy_eyelid", "feeling_cold", "decreased_appetite", "excessive_appetite",
    "excessive_anger", "loss_of_sensation", "focal_weakness", "slurring_words",
    "symptoms_of_the_face", "disturbance_of_memory", "paresthesia", "side_pain",
    "fever", "shoulder_pain", "shoulder_stiffness_or_tightness", "shoulder_weakness",
    "arm_cramps_or_spasms", "shoulder_swelling", "tongue_lesions", "leg_cramps_or_spasms",
    "abnormal_appearing_tongue", "ache_all_over", "lower_body_pain", "problems_during_pregnancy",
    "spotting_or_bleeding_during_pregnancy", "cramps_and_spasms", "upper_abdominal_pain",
    "stomach_bloating", "changes_in_stool_appearance", "unusual_color_or_odor_to_urine",
    "kidney_mass", "swollen_abdomen", "symptoms_of_prostate", "leg_stiffness_or_tightness",
    "difficulty_breathing", "rib_pain", "joint_pain", "muscle_stiffness_or_tightness",
    "pallor", "hand_or_finger_lump_or_mass", "chills", "groin_pain", "fatigue",
    "abdominal_distention", "regurgitation", "symptoms_of_the_kidneys", "melena",
    "flushing", "coughing_up_sputum", "seizures", "delusions_or_hallucinations",
    "shoulder_cramps_or_spasms", "joint_stiffness_or_tightness", "pain_or_soreness_of_breast",
    "excessive_urination_at_night", "bleeding_from_eye", "rectal_bleeding", "constipation",
    "temper_problems", "coryza", "wrist_weakness", "eye_strain", "hemoptysis", "lymphedema",
    "skin_on_leg_or_foot_looks_infected", "allergic_reaction", "congestion_in_chest",
    "muscle_swelling", "pus_in_urine", "abnormal_size_or_shape_of_ear", "low_back_weakness",
    "sleepiness", "apnea", "abnormal_breathing_sounds", "excessive_growth", "elbow_cramps_or_spasms",
    "feeling_hot_and_cold", "blood_clots_during_menstrual_periods", "absence_of_menstruation",
    "pulling_at_ears", "gum_pain", "redness_in_ear", "fluid_retention", "flu_like_syndrome",
    "sinus_congestion", "painful_sinuses", "fears_and_phobias", "recent_pregnancy",
    "uterine_contractions", "burning_chest_pain", "back_cramps_or_spasms", "stiffness_all_over",
    "muscle_cramps_contractures_or_spasms", "low_back_cramps_or_spasms", "back_mass_or_lump",
    "nosebleed", "long_menstrual_periods", "heavy_menstrual_flow", "unpredictable_menstruation",
    "painful_menstruation", "infertility", "frequent_menstruation", "sweating", "mass_on_eyelid",
    "swollen_eye", "eyelid_swelling", "eyelid_lesion_or_rash", "unwanted_hair", "symptoms_of_bladder",
    "irregular_appearing_nails", "itching_of_skin", "hurts_to_breathe", "nailbiting",
    "skin_dryness_peeling_scaliness_or_roughness", "skin_on_arm_or_hand_looks_infected",
    "skin_irritation", "itchy_scalp", "hip_swelling", "incontinence_of_stool",
    "foot_or_toe_cramps_or_spasms", "warts", "bumps_on_penis", "too_little_hair",
    "foot_or_toe_lump_or_mass", "skin_rash", "mass_or_swelling_around_the_anus",
    "low_back_swelling", "ankle_swelling", "hip_lump_or_mass", "drainage_in_throat",
    "dry_or_flaky_scalp", "premenstrual_tension_or_irritability", "feeling_hot",
    "feet_turned_in", "foot_or_toe_stiffness_or_tightness", "pelvic_pressure",
    "elbow_swelling", "elbow_stiffness_or_tightness", "early_or_late_onset_of_menopause",
    "mass_on_ear", "bleeding_from_ear", "hand_or_finger_weakness", "low_self_esteem",
    "throat_irritation", "itching_of_the_anus", "swollen_or_red_tonsils", "irregular_belly_button",
    "swollen_tongue", "lip_sore", "vulvar_sore", "hip_stiffness_or_tightness", "mouth_pain",
    "arm_weakness", "leg_lump_or_mass", "disturbance_of_smell_or_taste", "discharge_in_stools",
    "penis_pain", "loss_of_sex_drive", "obsessions_and_compulsions", "antisocial_behavior",
    "neck_cramps_or_spasms", "pupils_unequal", "poor_circulation", "thirst", "sleepwalking",
    "skin_oiliness", "sneezing", "bladder_mass", "knee_cramps_or_spasms", "premature_ejaculation",
    "leg_weakness", "posture_problems", "bleeding_in_mouth", "tongue_bleeding",
    "change_in_skin_mole_size_or_color", "penis_redness", "penile_discharge", "shoulder_lump_or_mass",
    "polyuria", "cloudy_eye", "hysterical_behavior", "arm_lump_or_mass", "nightmares",
    "bleeding_gums", "pain_in_gums", "bedwetting", "diaper_rash", "lump_or_mass_of_breast",
    "vaginal_bleeding_after_menopause", "infrequent_menstruation", "mass_on_vulva",
    "jaw_pain", "itching_of_scrotum", "postpartum_problems_of_the_breast", "eyelid_retracted",
    "hesitancy", "elbow_lump_or_mass", "muscle_weakness", "throat_redness", "joint_swelling",
    "tongue_pain", "redness_in_or_around_nose", "wrinkles_on_skin", "foot_or_toe_weakness",
    "hand_or_finger_cramps_or_spasms", "back_stiffness_or_tightness", "wrist_lump_or_mass",
    "skin_pain", "low_back_stiffness_or_tightness", "low_urine_output", "skin_on_head_or_neck_looks_infected",
    "stuttering_or_stammering", "problems_with_orgasm", "nose_deformity", "lump_over_jaw",
    "sore_in_nose", "hip_weakness", "back_swelling", "ankle_stiffness_or_tightness", "ankle_weakness",
    "neck_weakness"
]


# Generate the CREATE TABLE statement
table_name = "MachineLearnHealth"
sql_create_table = f"CREATE TABLE {table_name} (\n"
for i, col in enumerate(columns):
    if i == 0:  # First column (Disease) is a String
        sql_create_table += f"    `{col.replace(' ', '_')}` String,\n"
    else:  # All other columns (symptoms) are UInt8
        sql_create_table += f"    `{col.replace(' ', '_')}` UInt8,\n"
sql_create_table = sql_create_table.rstrip(",\n")  # Remove the last comma
sql_create_table += "\n) ENGINE = MergeTree()\nORDER BY `disease`;"

# Print the SQL statement
print(sql_create_table)

# Optionally, save the query to a file
with open("create_table.sql", "w") as f:
    f.write(sql_create_table)
