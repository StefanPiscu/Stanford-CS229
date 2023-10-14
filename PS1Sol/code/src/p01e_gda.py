import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)


    # *** START CODE HERE ***
    model=GDA()
    model.fit(x_train, y_train)

    util.plot(x_train, y_train, model.theta, "output/p01e_{}.png".format(pred_path[-5]))
    x_eval, y_eval=util.load_dataset(eval_path, add_intercept=True)
    y_pred=model.predict(x_eval)

    np.savetxt(pred_path, y_pred>0.5, fmt='%d')
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***

        m, n = x.shape
        self.theta=np.zeros(n+1)

        y_mean= (y==1).sum()/m
        x0_mean=np.sum(x[y==0], axis=0)/((y==0).sum())
        x1_mean=np.sum(x[y==1], axis=0)/((y==1).sum())
        covariance=((x[y==0]-x0_mean).T.dot((x[y==0]-x0_mean))+(x[y==1]-x1_mean).T.dot((x[y==1]-x1_mean)))/m
        covariance_inv=np.linalg.inv(covariance)

        self.theta[0]=(x0_mean+x1_mean).T.dot(covariance_inv).dot(x0_mean-x1_mean)/2-np.log((1-y_mean)/y_mean)
        self.theta[1:]=covariance_inv.dot(x1_mean-x0_mean)
    
        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-np.matmul(x, self.theta)))
        # *** END CODE HERE
