import numpy as np
import pandas as pd
from bokeh.io import show as display
from bokeh.models import ColorBar, ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import linear_cmap

from .colors import PALETTE_FUNC
from .plot import boxplot, scatter
from .utils import _output


def features_importance(
    shap_values, features, output_file_name="features_importance.html", show=True
):  # pragma: no cover
    """Display the features importance of the model given the shap values

    Params:
    -------
    shap_values (np.ndarray): the shap values for the model and the data
    features (pd.DataFrame or [str]): the features used by the model
    output_file_name (str): the name of the file to output the figure
    show (bool): whether or not to display the figure
    """
    _output(f"{output_file_name}")

    feature_names = features.columns.values

    correlation = np.array(
        [
            np.sign(np.corrcoef(shap_values[:, i], features.iloc[:, i])[0, 1])
            for i in range(len(feature_names))
        ]
    )

    mean_abs_shap_values = np.sum(np.abs(shap_values), axis=0) / shap_values.shape[0]
    order_idx = np.argsort(mean_abs_shap_values[np.where(mean_abs_shap_values > 0)])

    feature_names_ordered = feature_names[order_idx]
    mean_abs_shap_values_ordered = mean_abs_shap_values[order_idx]
    correlation_ordered = correlation[order_idx]

    source = ColumnDataSource(
        data=dict(
            values=mean_abs_shap_values_ordered,
            features=feature_names_ordered,
            correlation=correlation_ordered,
        )
    )

    y_range = (0, np.max(mean_abs_shap_values_ordered) * 1.1)

    fig = figure(
        x_range=y_range,
        y_range=feature_names_ordered,
        plot_height=75 * len(feature_names_ordered),
        plot_width=1000,
        title="Features importance",
        toolbar_location=None,
        tools="",
    )

    color_mapper = linear_cmap("correlation", PALETTE_FUNC(2), -1, 1)

    fig.hbar(
        right="values", y="features", height=0.5, source=source, color=color_mapper
    )

    color_bar = ColorBar(
        color_mapper=color_mapper["transform"],
        border_line_color=None,
        location=(0, 0),
        width=8,
        major_label_overrides={-1: "", -0.5: "Negative", 0: "", 0.5: "Positive", 1: ""},
        major_label_text_align="center",
        major_label_text_font_size="11px",
        label_standoff=15,
        title="Correlation",
        title_text_align="center",
        title_text_font_size="11px",
        title_standoff=10,
        padding=30,
    )

    fig.add_layout(color_bar, "right")
    fig.xgrid.grid_line_color = None
    fig.xaxis.axis_label = "Mean Shap value"

    if show:
        display(fig)

    return fig


def feature_importance_dependence(
    shap_values,
    features,
    X,
    Z=None,
    output_file_name="features_importance_dependence.html",
    show=True,
):  # pragma: no cover
    """Display the data showing the Shap values for a given variable X in function of another
    variable Z. The idea is to try to understand if a protected variable Z impact the shap value
    of X. If it does the model is likely to be biased.

    Params:
    -------
    shap_values (np.ndarray): the shap values for the model and the data
    features (pd.DataFrame or [str]): the features used by the model
    X (str or pd.Series): the name of the feature to investigate
    Z (str or pd.Series): the name of the protected variable
    output_file_name (str): the name of the file to output the figure
    show (bool): whether or not to display the figure
    """
    if isinstance(X, pd.Series):
        X = X.name
    x_shap_values = pd.DataFrame(shap_values, columns=features.columns)[X].rename(
        "Shap values"
    )

    if features[X].nunique() <= 10:
        if isinstance(Z, pd.Series):
            Z = Z.name

        if Z and Z != X:
            X = [X, Z]

        return boxplot(x_shap_values, X, data=features)

    return scatter(
        X,
        x_shap_values,
        Z=Z,
        data=features,
        output_file_name=output_file_name,
        show=show,
    )
