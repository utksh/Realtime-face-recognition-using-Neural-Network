find ./input_dir -name ".DS_Store" -delete
python3 aligndata_first.py
python3 create_classifier_se.py
#python3 main_face_detection.py
