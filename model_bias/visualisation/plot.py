from collections.abc import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from bokeh.io import show as display
from bokeh.layouts import gridplot
from bokeh.models import (
    ColorBar,
    ColumnDataSource,
    FactorRange,
    LabelSet,
    Legend,
    Range1d,
)
from bokeh.plotting import figure
from bokeh.transform import cumsum, factor_cmap, linear_cmap

from .colors import (
    BOTTOM_BOX_COLOR,
    LINE_COLOR,
    PALETTE_FUNC,
    PALETTE_LARGE,
    TOP_BOX_COLOR,
)
from .utils import _output, _retrieve_value

## Categorical frequencies


def _counts(
    X, data=None, output_file_name="value_counts.html", show=True
):  # pragma: no cover
    """ Plot the categories and the proportion of individuals in each category

    Params:
    -------
    X (pd.Series or str): the data to plot
    data (pd.DataFrame): if `X` is str the data to retrieve `X` from
    output_file_name (str): if not in a Jupyter notebook the plot will be exported to this file
    show (bool): whether the plot should be shown
    """
    X = _retrieve_value(X, data=data).astype(str)
    _output(f"{X.name}_{output_file_name}")

    values, val_counts = [np.array(d) for d in zip(*X.value_counts().to_dict().items())]

    order_idx = np.argsort(val_counts)
    values = values[order_idx]
    val_counts = val_counts[order_idx]

    percentages = [f"{p}%" for p in (val_counts / np.sum(val_counts) * 100).round(2)]
    source = ColumnDataSource(
        data=dict(
            values=values,
            counts=val_counts,
            percentages=percentages,
            color=PALETTE_FUNC(len(values)),
        )
    )

    y_range = (0, np.max(val_counts) * 1.1)

    fig = figure(
        x_range=y_range,
        y_range=values,
        plot_height=75 * len(val_counts),
        plot_width=1000,
        title=X.name.title(),
        toolbar_location=None,
        tools="",
    )

    fig.hbar(right="counts", y="values", height=0.5, color="color", source=source)

    text_props = {
        "source": source,
        "text_align": "center",
        "text_baseline": "alphabetic",
        "text_font_style": "bold",
        "text_font_size": "12px",
    }
    fig.text(
        x="counts",
        y="values",
        y_offset=5,
        x_offset=25,
        text="percentages",
        **text_props,
    )

    fig.ygrid.grid_line_color = None

    if show:
        display(fig)

    return fig


def _counts_by(
    X, Z, data=None, output_file_name="value_counts.html", show=True
):  # pragma: no cover
    """ Plot the categories and the proportion of individuals in each category

    Params:
    -------
    X (pd.Series or str): the data to plot
    Z (pd.Series or str): the variable used to split the data in categories
    data (pd.DataFrame): if `X` is str the data to retrieve `X` from
    output_file_name (str): if not in a Jupyter notebook the plot will be exported to this file
    """
    X = _retrieve_value(X, data=data).astype(str)
    Z = _retrieve_value(Z, data=data).astype(str)
    _output(f"{X.name}_by_{Z.name}_{output_file_name}")

    x_cats = sorted(Z.unique())
    y_cats = sorted(X.unique())

    values = [(x_cat, y_cat) for x_cat in x_cats for y_cat in y_cats]
    cat_counts = sum(
        [
            [data[(Z == x_cat) & (X == y_cat)].shape[0] for y_cat in y_cats]
            for x_cat in x_cats
        ],
        [],
    )
    cat_percentages = sum(
        [
            [
                data[(Z == x_cat) & (X == y_cat)].shape[0] / data[(Z == x_cat)].shape[0]
                for y_cat in y_cats
            ]
            for x_cat in x_cats
        ],
        [],
    )
    cat_percentages = [f"{round(p*100, 2)}%" for p in cat_percentages]

    source = ColumnDataSource(
        data=dict(values=values, counts=cat_counts, percentages=cat_percentages)
    )

    y_range = (0, np.max(cat_counts) * 1.1)

    fig = figure(
        x_range=y_range,
        y_range=FactorRange(*values),
        plot_height=50 * len(values),
        plot_width=1000,
        title=X.name.title(),
        toolbar_location=None,
        tools="",
    )

    fig.hbar(
        y="values",
        right="counts",
        height=0.5,
        source=source,
        color=None,
        fill_color=factor_cmap(
            "values", palette=PALETTE_FUNC(len(y_cats)), factors=y_cats, start=1, end=2
        ),
    )

    text_props = {
        "source": source,
        "text_align": "center",
        "text_baseline": "alphabetic",
        "text_font_style": "bold",
        "text_font_size": "12px",
    }
    fig.text(
        y="values",
        x="counts",
        y_offset=5,
        x_offset=25,
        text="percentages",
        **text_props,
    )

    fig.ygrid.grid_line_color = None

    if show:
        display(fig)

    return fig


def counts(
    X, Z=None, data=None, output_file_name="value_counts.html", show=True
):  # pragma: no cover
    """ Plot the categories and the proportion of individuals in each category

    Params:
    -------
    X (pd.Series or str): the data to plot
    Z (pd.Series or str): the variable used to split the data in categories
    data (pd.DataFrame): if `X` is str the data to retrieve `X` from
    output_file_name (str): if not in a Jupyter notebook the plot will be exported to this file
    """
    if Z is None:
        return _counts(X, data=data, output_file_name=output_file_name)
    return _counts_by(X, Z, data=data, output_file_name=output_file_name, show=show)


## Distribution


def distribution(
    X, data=None, output_file_name="distribution.html", show=True
):  # pragma: no cover
    """Plot the distribition of a continuous variable

    Params:
    -------
    X (pd.Series or str): the data to plot
    data (pd.DataFrame): if `X` is str the data to retrieve `X` from
    output_file_name (str): if not in a Jupyter notebook the plot will be exported to this file
    """
    X = _retrieve_value(X, data=data)
    _output(f"{X.name}_{output_file_name}")

    hist, edges = np.histogram(X, bins="auto", density=True)
    source = ColumnDataSource(
        data=dict(
            top=hist,
            bottom=np.zeros(len(hist)),
            left=edges[:-1],
            right=edges[1:],
            color=PALETTE_FUNC(len(edges[:-1])),
        )
    )

    fig = figure(
        plot_height=250,
        plot_width=1000,
        title=X.name.title(),
        toolbar_location=None,
        tools="",
    )

    fig.quad(
        top="top",
        bottom="bottom",
        left="left",
        right="right",
        color="color",
        source=source,
        legend_label="Histogram",
    )

    try:
        kde = sm.nonparametric.KDEUnivariate(X)
        kde.fit()
        fig.line(
            x=kde.support,
            y=kde.density,
            line_width=2,
            color=LINE_COLOR,
            legend_label="Density",
        )
    except Exception:
        pass

    fig.legend.click_policy = "hide"

    if show:
        display(fig)

    return fig


## Boxplots


def _plot_boxplot(
    X, data=None, output_file_name="boxplot.html", show=True
):  # pragma: no cover
    """Plot the boxplot associated with a continuous variable

    Params:
    -------
    X (pd.Series or str): the data to plot
    data (pd.DataFrame): if `X` is str the data to retrieve `X` from
    output_file_name (str): if not in a Jupyter notebook the plot will be exported to this file
    """
    X = _retrieve_value(X, data=data)
    _output(f"{X.name}_{output_file_name}")

    q1 = X.quantile(q=0.25)
    q2 = X.quantile(q=0.5)
    q3 = X.quantile(q=0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr

    out = X[(X > upper) | (X < lower)]

    source = None
    if not out.empty:
        outx = []
        outy = []
        for keys in out.index:
            outx.append("")
            outy.append(out.loc[keys])

        source = ColumnDataSource(data=dict(y=outy, x=outx))

    fig = figure(
        x_range=[""],
        plot_height=500,
        plot_width=500,
        title=X.name.title(),
        tools="",
        toolbar_location=None,
    )

    qmin = X.quantile(q=0.00)
    qmax = X.quantile(q=1.00)
    upper = min(upper, qmax)
    lower = max(lower, qmin)

    # stems
    fig.segment(x0=[""], y0=upper, x1=[""], y1=q3, line_color="black")
    fig.segment(x0=[""], y0=lower, x1=[""], y1=q1, line_color="black")

    # boxes
    fig.vbar(
        x=[""],
        width=0.15,
        top=q2,
        bottom=q3,
        fill_color=TOP_BOX_COLOR,
        line_color="black",
    )
    fig.vbar(
        x=[""],
        width=0.15,
        top=q1,
        bottom=q2,
        fill_color=BOTTOM_BOX_COLOR,
        line_color="black",
    )

    # whiskers (almost-0 height rects simpler than segments)
    fig.rect(x=[""], y=lower, width=0.05, height=0.01, line_color="black")
    fig.rect(x=[""], y=upper, width=0.05, height=0.01, line_color="black")

    # outliers
    if source:
        fig.circle(
            x="x",
            y="y",
            size=6,
            source=source,
            color=linear_cmap("y", PALETTE_LARGE, min(X), max(X)),
        )

    fig.xgrid.grid_line_color = None

    if show:
        display(fig)

    return fig


def _plot_boxplot_by(
    X, Z, data=None, output_file_name="boxplot.html", show=True
):  # pragma: no cover
    """Plot the boxplot associated with a continuous variable in function of
    another variable

    Params:
    -------
    X (pd.Series or str): the data to plot
    Z (pd.Series or str): the categorical variable to use to display the boxplots by
    data (pd.DataFrame): if `X` is str the data to retrieve `X` from
    output_file_name (str): if not in a Jupyter notebook the plot will be exported to this file
    """
    X = _retrieve_value(X, data=data)
    Z = _retrieve_value(Z, data=data).astype(str)

    if len(Z.shape) > 1:
        if Z.shape[1] > 2:
            raise ValueError("Only 2 levels can be displayed")
    else:
        Z = pd.DataFrame({Z.name: Z})

    agg_str = "_".join(Z.columns)
    _output(f"{X.name}_by_{agg_str}_{output_file_name}")

    dict_data = {X.name: X}
    dict_data.update({c: Z[c] for c in Z.columns})
    data = pd.DataFrame(dict_data)

    groups = data.groupby(list(Z.columns), sort=True)
    cats = list(groups.groups.keys())
    q1 = groups.quantile(q=0.25)
    q2 = groups.quantile(q=0.5)
    q3 = groups.quantile(q=0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr

    def outliers(group):
        cat = group.name
        return group[
            (group[X.name] > upper.loc[cat][X.name])
            | (group[X.name] < lower.loc[cat][X.name])
        ][X.name]

    out = groups.apply(outliers).dropna()

    source = None
    if not out.empty:
        outx = []
        outy = []
        for keys in out.index:
            outx.append(keys[:-1])
            outy.append(out.loc[keys[:-1]].loc[keys[-1]])

        source = ColumnDataSource(data=dict(y=outy, cat=outx))

    min_X, max_X = min(X), max(X)
    if abs(min_X - max_X) < 0.01:
        y_range = (-0.01, 0.01)
    else:
        y_range = (min_X * 1.1, max_X * 1.1)

    fig = figure(
        x_range=cats if not isinstance(cats[0], Iterable) else FactorRange(*cats),
        y_range=y_range,
        plot_height=750,
        plot_width=1000,
        title=X.name.title(),
        tools="",
        toolbar_location=None,
    )

    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)
    upper[X.name] = [
        min([x, y]) for (x, y) in zip(list(qmax.loc[:, X.name]), upper[X.name])
    ]
    lower[X.name] = [
        max([x, y]) for (x, y) in zip(list(qmin.loc[:, X.name]), lower[X.name])
    ]

    # stems
    fig.segment(x0=cats, y0=upper[X.name], x1=cats, y1=q3[X.name], line_color="black")
    fig.segment(x0=cats, y0=lower[X.name], x1=cats, y1=q1[X.name], line_color="black")

    # boxes
    fig.vbar(
        x=cats,
        width=0.25,
        top=q2[X.name],
        bottom=q3[X.name],
        fill_color=TOP_BOX_COLOR,
        line_color="black",
    )
    fig.vbar(
        x=cats,
        width=0.25,
        top=q1[X.name],
        bottom=q2[X.name],
        fill_color=BOTTOM_BOX_COLOR,
        line_color="black",
    )

    # whiskers (almost-0 height rects simpler than segments)
    fig.rect(
        x=cats,
        y=upper[X.name],
        width=0.1,
        height=0.01 * abs(y_range[0] - y_range[1]),
        fill_color=TOP_BOX_COLOR,
        line_color="black",
    )
    fig.rect(
        x=cats,
        y=lower[X.name],
        width=0.1,
        height=0.01 * abs(y_range[0] - y_range[1]),
        fill_color=BOTTOM_BOX_COLOR,
        line_color="black",
    )

    # outliers
    if source:
        fig.circle(
            x="cat",
            y="y",
            size=6,
            source=source,
            color=linear_cmap("y", PALETTE_LARGE, min_X, max_X),
        )

    fig.xgrid.grid_line_color = None

    if show:
        display(fig)

    return fig


def boxplot(
    X, Z=None, data=None, output_file_name="boxplot.html", show=True
):  # pragma: no cover
    """Plot the boxplot associated with a variable

    Params:
    -------
    X (pd.Series or str): the data to plot
    Z (pd.Series or str or [str] or [pd.Series]): Default: None. the categorical variable
                                                    to use to display the boxplots by
    data (pd.DataFrame): if `X` is str the data to retrieve `X` from
    output_file_name (str): if not in a Jupyter notebook the plot will be exported to this file
    """

    if Z is None:
        return _plot_boxplot(X, data, output_file_name)

    return _plot_boxplot_by(X, Z, data, output_file_name, show=show)


## Scatter


def scatter(
    X, Y, Z=None, data=None, output_file_name="scatter.html", show=True
):  # pragma: no cover
    """Scatter plot 2 dimensions

    Params:
    -------
    X (pd.Series or str): the x coordinates
    Y (pd.Series or str): the y coordinates
    Z (pd.Series or str): the variable that will be used for colouring
    data (pd.DataFrame): if `X` is str the data to retrieve `X` from
    output_file_name (str): if not in a Jupyter notebook the plot will be exported to this file
    """

    X = _retrieve_value(X, data=data)
    Y = _retrieve_value(Y, data=data)
    _output(f"{X.name}_{Y.name}_{output_file_name}")

    color_values = _retrieve_value(Z, data=data) if Z else X * Y
    unique_color_values = sorted(color_values.unique())
    if len(unique_color_values) > 256:
        palette = PALETTE_LARGE
        major_label_overrides = {}
        min_color_value = np.min(color_values)
        max_color_value = np.max(color_values)
    else:
        palette = PALETTE_FUNC(len(unique_color_values))
        color_values = color_values.apply(unique_color_values.index)
        major_label_overrides = {
            unique_color_values.index(v) + 0.5: str(v) for v in unique_color_values
        }
        major_label_overrides.update(
            {i: "" for i in range(len(unique_color_values) + 1)}
        )
        min_color_value = np.min(color_values)
        max_color_value = np.max(color_values) + 1

    # create the scatter plot
    fig = figure(
        plot_height=600,
        plot_width=750,
        min_border=10,
        min_border_left=50,
        tools="",
        toolbar_location=None,
        x_axis_location=None,
        y_axis_location=None,
    )

    slope, intercept = np.polyfit(X, Y, 1, full=True)[0]
    y_regression = [slope * i + intercept for i in X]

    source = ColumnDataSource(
        data=dict(x=X, y=Y, y_regression=y_regression, color_val=color_values)
    )

    color_mapper = linear_cmap("color_val", palette, min_color_value, max_color_value)

    fig.scatter(x="x", y="y", source=source, size=3, color=color_mapper)

    fig.line(
        x="x",
        y="y_regression",
        color=LINE_COLOR,
        line_width=2.0,
        legend_label=f"y = {slope:0.2f}x {intercept:+0.2f}",
        source=source,
    )

    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"

    if Z and _retrieve_value(Z, data=data).name != X.name:
        color_bar = ColorBar(
            color_mapper=color_mapper["transform"],
            border_line_color=None,
            location=(0, 0),
            width=8,
            label_standoff=15,
            title=Z,
            title_text_align="center",
            title_text_font_size="11px",
            title_standoff=10,
            major_label_overrides=major_label_overrides,
            major_label_text_align="center",
            major_label_text_font_size="11px",
            padding=30,
        )
        fig.add_layout(color_bar, "right")

    # create the horizontal histogram
    hhist, hedges = np.histogram(X, bins="auto")
    hmax = max(hhist) * 1.1
    hsource = ColumnDataSource(
        data=dict(
            bottom=np.zeros(len(hedges) - 1),
            top=hhist,
            left=hedges[:-1],
            right=hedges[1:],
            color=PALETTE_FUNC(len(hedges[:-1])),
        )
    )

    ph = figure(
        plot_width=fig.plot_width,
        plot_height=200,
        x_range=fig.x_range,
        min_border=10,
        min_border_left=50,
        y_axis_location="right",
        tools="",
        toolbar_location=None,
    )
    ph.xgrid.grid_line_color = None
    ph.yaxis.major_label_orientation = np.pi / 4
    ph.y_range = Range1d(0, hmax)
    ph.xaxis.axis_label = X.name

    hh = ph.quad(
        bottom="bottom",
        left="left",
        right="right",
        top="top",
        color="color",
        source=hsource,
    )

    hlegend_items = [("Histogram", [hh])]

    try:
        kde = sm.nonparametric.KDEUnivariate(X)
        kde.fit()
        ph.extra_y_ranges = {
            "hdensity": Range1d(start=0, end=np.max(kde.density) * 1.1)
        }
        hl = ph.line(
            x=kde.support,
            y=kde.density,
            line_width=2,
            color=LINE_COLOR,
            y_range_name="hdensity",
        )
        hlegend_items.append(("Density", [hl]))
    except Exception:
        pass

    hlegend = Legend(items=hlegend_items, location="center")
    hlegend.click_policy = "hide"

    ph.add_layout(hlegend, "right")

    # create the vertical histogram
    vhist, vedges = np.histogram(Y, bins="auto")
    vmax = max(vhist) * 1.1
    vsource = ColumnDataSource(
        data=dict(
            bottom=vedges[:-1],
            top=vedges[1:],
            left=np.zeros(len(vedges) - 1),
            right=vhist,
            color=PALETTE_FUNC(len(vedges[:-1])),
        )
    )

    pv = figure(
        plot_width=200,
        plot_height=fig.plot_height,
        y_range=fig.y_range,
        min_border=10,
        y_axis_location="left",
        toolbar_location=None,
        tools="",
    )
    pv.ygrid.grid_line_color = None
    pv.xaxis.major_label_orientation = np.pi / 4
    pv.x_range = Range1d(vmax, 0)
    pv.yaxis.axis_label = Y.name

    hv = pv.quad(
        left="left",
        bottom="bottom",
        top="top",
        right="right",
        color="color",
        source=vsource,
    )

    vlegend_items = [("Histogram", [hv])]

    try:
        kde = sm.nonparametric.KDEUnivariate(Y)
        kde.fit()
        pv.extra_x_ranges = {"vdensity": Range1d(np.max(kde.density) * 1.1, 0)}
        vl = pv.line(
            x=kde.density,
            y=kde.support,
            line_width=2,
            color=LINE_COLOR,
            x_range_name="vdensity",
        )
        vlegend_items.append(("Density", [vl]))
    except Exception:
        pass

    vlegend = Legend(items=vlegend_items, location="center")
    vlegend.click_policy = "hide"
    vlegend.orientation = "horizontal"

    pv.add_layout(vlegend, "above")

    layout = gridplot([[pv, fig], [None, ph]], merge_tools=False)

    if show:
        display(layout)

    return layout


def pie_proportion(
    percentage,
    target=None,
    label=None,
    title="",
    output_file_name="pie_proportion.html",
    show=True,
):  # pragma: no cover
    """Plot the percentage in a pie format.

    Params:
    -------
    percentage (float): the proportion to plot
    target (float): the proportion to reach
    label (str): the label to add to the plot
    title (str): the title of the plot
    """
    _output(f"{title}_{output_file_name}")

    fig = figure(
        plot_height=350,
        plot_width=1000,
        x_range=(-0.5, 0.5),
        y_range=(0.5 if max([percentage, target or 0.0]) <= 0.5 else 0, 2),
        title=title,
        toolbar_location=None,
        tools="",
    )

    if target is not None and target > percentage:
        fig.annular_wedge(
            x=0,
            y=1,
            inner_radius=0.075,
            outer_radius=0.15,
            start_angle=0,
            end_angle=np.radians(target * 360),
            line_color=LINE_COLOR,
            line_width=2.5,
            fill_alpha=0.0,
        )
    fig.annular_wedge(
        x=0,
        y=1,
        inner_radius=0.075,
        outer_radius=0.15,
        start_angle=0,
        end_angle=np.radians(percentage * 360),
        line_color=PALETTE_FUNC(2)[0],
        line_width=2.5,
        fill_color=PALETTE_FUNC(2)[0],
    )
    if target is not None and target < percentage:
        fig.annular_wedge(
            x=0,
            y=1,
            inner_radius=0.075,
            outer_radius=0.15,
            start_angle=0,
            end_angle=np.radians(target * 360),
            line_color=LINE_COLOR,
            line_width=2.5,
            fill_alpha=0.0,
        )

    fig.axis.axis_label = None
    fig.axis.visible = False
    fig.grid.grid_line_color = None

    labels = LabelSet(
        x=0,
        y=1,
        text=[f"{percentage*100:0.2f}%"] if label is None else [f"{label}"],
        text_align="center",
        text_baseline="middle",
    )
    fig.add_layout(labels)

    if show:
        display(fig)

    return fig


def pie(X, data=None, output_file_name="pie.html", show=True):  # pragma: no cover
    """ Plot a pie diagram with the proportions of the values in the series

    Params:
    -------
    X (pd.Series or str): the data to plot porportion for
    data (pd.DataFrame): the dataframe to collect the series from
    """
    X = _retrieve_value(X, data=data)
    _output(f"{X.name}_{output_file_name}")

    values, val_counts = [
        np.array(d) for d in zip(*X.value_counts(sort=False).to_dict().items())
    ]
    percentages = [f"{p}%" for p in (val_counts / np.sum(val_counts) * 100).round(2)]
    source = ColumnDataSource(
        data=dict(
            values=values,
            counts=val_counts,
            percentages=percentages,
            angle=val_counts / np.sum(val_counts) * 2 * np.pi,
            color=PALETTE_FUNC(len(values)),
        )
    )

    fig = figure(
        plot_height=350,
        plot_width=1000,
        x_range=(-0.5, 0.5),
        title=X.name.title(),
        toolbar_location=None,
        tools="",
    )

    fig.annular_wedge(
        x=0,
        y=1,
        inner_radius=0.075,
        outer_radius=0.15,
        start_angle=cumsum("angle", include_zero=True),
        end_angle=cumsum("angle"),
        line_color="white",
        fill_color="color",
        legend_field="values",
        source=source,
    )

    fig.axis.axis_label = None
    fig.axis.visible = False
    fig.grid.grid_line_color = None

    if show:
        display(fig)

    return fig
