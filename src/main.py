from data_preprocessing import process_data
from adaboost import AdaBoost
from evaluation import compute_accuracy, compute_confusion_matrix, compute_auc
from visualization import plot_comparison_weak_vs_boost
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def main():
    # load and preprocess data
    print("Processing data...")
    X_train_full, X_test, y_train_full, y_test = process_data()
    print("Data shapes:")
    print("  X_train:", X_train_full.shape, "X_test:", X_test.shape)

    # split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
    print("Training shape:", X_train.shape, "Validation shape:", X_val.shape)

    # train adaboost model on training data
    n_rounds = 50
    learning_rate = 0.1
    model = AdaBoost(n_rounds = n_rounds, learning_rate=learning_rate)
    print("Training AdaBoost model...")
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # predict using the trained model
    print("Making predictions...")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # get raw prediction scores for AUC computation
    train_scores = model.predict_scores(X_train)
    test_scores = model.predict_scores(X_test)

    # evaluate model performance
    train_accuracy = compute_accuracy(y_train, train_pred)
    test_accuracy = compute_accuracy(y_test, test_pred)
    print("Training accuracy:", train_accuracy)
    print("Test accuracy:", test_accuracy)

    #           predicted neg    predicted positive
    #  True neg    |  TN      |      FP
    #  True pos    |  FN      |      TP 
    print("Confusion Matrix (Test):")
    print(compute_confusion_matrix(y_test, test_pred))

    train_auc = compute_auc(y_train, train_scores)
    test_auc = compute_auc(y_test, test_scores)
    print("Training AUC:", train_auc)
    print("Testing AUC:", test_auc)

    # train a single weak learner to obtain baseline raw prediction scores
    weak_learner = DecisionTreeClassifier(max_depth=1, random_state=42)
    # use fit without sample weights because the weak learner is not reweighted
    weak_learner.fit(X_train, y_train)
    weak_train_pred = weak_learner.predict(X_train)
    weak_test_pred = weak_learner.predict(X_test)
    weak_train_accuracy = compute_accuracy(y_train, weak_train_pred)
    weak_test_accuracy = compute_accuracy(y_test, weak_test_pred)
    print("Weak learner training accuracy:", weak_train_accuracy)
    print("Weak learner test accuracy:", weak_test_accuracy)
    weak_test_scores = weak_learner.predict_proba(X_test)[:, 1]

    # ROC comparison between weak learner and boosted model
    plot_comparison_weak_vs_boost(y_test, weak_test_scores, test_scores)

if __name__ == "__main__":
    main()