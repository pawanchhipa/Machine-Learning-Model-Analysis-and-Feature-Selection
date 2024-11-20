import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def plot_parity(y_cv_test, y_pred_test, y_cv_train=None, y_pred_train=None, label=None, ylim=[50,900]):
    """
    Function to make parity plots.
    
    Parameters:
    -----------
    y_cv_test : array-like
        True test values
    y_pred_test : array-like
        Predicted test values
    y_cv_train : array-like, optional
        True training values
    y_pred_train : array-like, optional
        Predicted training values
    label : str, optional
        Label for the plot
    ylim : list, optional
        Y-axis limits [min, max]
    """
    
    # Plot Parity plot
    rmse_test = np.sqrt(mean_squared_error(y_cv_test, y_pred_test))
    r2_test = r2_score(y_cv_test, y_pred_test)

    if y_cv_train is None:
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5,4), sharey=True, sharex=True)
    else:
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9,4), sharey=True, sharex=True)

    ax1.scatter(y_cv_test, y_pred_test)
    ax1.text(0.95, 0.26, label, transform=ax1.transAxes, ha='right', fontsize=14)
    ax1.text(0.95, 0.18, "RMSE: %.2f"%rmse_test, transform=ax1.transAxes, ha='right', fontsize=14)
    ax1.text(0.95, 0.1, "R$^2$: %.2f"%r2_test, transform=ax1.transAxes, ha='right', fontsize=14)
    ax1.plot(ylim, ylim, '--k')
    ax1.set_xlabel('True y', fontsize=14)
    ax1.set_ylabel('Pred y', fontsize=14)
    ax1.set_xlim(ylim[0], ylim[1])
    ax1.set_ylim(ylim[0], ylim[1])

    if y_cv_train is not None:
        rmse_train = np.sqrt(mean_squared_error(y_cv_train, y_pred_train))
        r2_train = r2_score(y_cv_train, y_pred_train)

        ax2.scatter(y_cv_train, y_pred_train, c='m')
        ax2.text(0.95, 0.26, "Train", transform=ax2.transAxes, ha='right', fontsize=14)
        ax2.text(0.95, 0.18, "RMSE: %.2f"%rmse_train, transform=ax2.transAxes, ha='right', fontsize=14)
        ax2.text(0.95, 0.1, "R2: %.2f"%r2_train, transform=ax2.transAxes, ha='right', fontsize=14)
        ax2.plot(ylim, ylim, '--k')

        ax2.set_xlabel('True y', fontsize=14)
        ax2.set_xlim(ylim[0], ylim[1])
        ax2.set_ylim(ylim[0], ylim[1])

    plt.tight_layout()
    plt.show()

    return None
