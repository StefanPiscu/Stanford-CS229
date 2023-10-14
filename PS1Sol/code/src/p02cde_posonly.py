import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels

    x_train, y_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    model_t=LogisticRegression()
    model_t.fit(x_train, y_train)
    
    util.plot(x_train, y_train, model_t.theta, "output/p02c.png")
    t_pred=model_t.predict(x_test)
    
    np.savetxt(pred_path_c, t_pred>0.5, fmt='%d')

    # Make sure to save outputs to pred_path_c
    # Part (d): Train on y-labels and test on true labels

    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)

    model_y=LogisticRegression()
    model_y.fit(x_train, y_train)

    util.plot(x_test, y_test, model_y.theta, "output/p02d.png")
    y_pred=model_y.predict(x_test)
    np.savetxt(pred_path_d, y_pred>0.5, fmt='%d')

    # Make sure to save outputs to pred_path_d
    # Part (e): Apply correction factor using validation set and test on true labels

    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    v_pred=model_y.predict(x_valid)
    alpha=np.mean(v_pred[y_valid==1])

    correction = 1 + np.log(2 / alpha - 1) / model_y.theta[0]
    util.plot(x_test, t_test, model_y.theta, "output/p02e.png", correction)

    t_pred=y_pred/alpha
    np.savetxt(pred_path_e, t_pred>0.5, fmt='%d')
    
    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE
