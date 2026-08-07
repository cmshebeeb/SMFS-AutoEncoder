from src.evaluation.metrics import clustering_accuracy, nmi_score


def test_perfect_prediction():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]

    assert clustering_accuracy(y_true, y_pred) == 1.0
    assert nmi_score(y_true, y_pred) == 1.0


def test_random_prediction():
    y_true = [0, 0, 1, 1]
    y_pred = [1, 0, 0, 1]

    assert clustering_accuracy(y_true, y_pred) < 1.0
    assert nmi_score(y_true, y_pred) < 1.0