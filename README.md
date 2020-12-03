# SemEval 2021 Task 8: MeasEval - Counts and Measurements
Counts and measurements are an important part of scientific discourse. It is relatively easy to find measurements in text, but a bare measurement like "17 mg" is not informative. However, relatively little attention has been given to parsing and extracting these important semantic relations. This is challenging because the way scientists write can be ambiguous and inconsistent, and the location of this information relative to the measurement can vary greatly.

MeasEval is a new entity and semantic relation extraction task focused on finding counts and measurements, attributes of these quantities, and additional information including measured entities, properties, and measurement contexts.


More details: https://competitions.codalab.org/competitions/25770

Dataset: https://github.com/harperco/MeasEval

# Project
This project was made using Python 3.7 and the dependencies necessary to run it are in the INSTALL.txt file.

In this project, the objective is to identify the Quantities, MeasuredEntities, and MeasuredProperties from the provided texts in the dataset. Features extracted from the text for the model include the lowercase form of a token, whether a token contains digits, the POS tag of a token, and the surrounding tokens. This model used conditional random fields (CRFs) with gradient descent using the L-BFGS method in the *sklearn_crfsuite* package. 