# from starforge_mult_search.analysis.figures.figure_preamble import *
# from bash_command import bash_command as bc
import astropy.constants as const
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.io as pio

import dash
from dash import dcc, html, Output, Input, State
import dash_bootstrap_components as dbc


pc=const.pc.cgs.value
au=const.au.cgs.value

unit = pc / au

##Preliminaries -- parsing data
###############################################################################
sink_cols = np.array(("t", "id", "px", "py", "pz", "vx", "vy", "vz", "h", "m"))
sink_cols = np.concatenate((sink_cols, ["sys_id", "mtot", "sma", "ecc"]))
mcol = np.where(sink_cols == "m")[0][0]
pxcol = np.where(sink_cols == "px")[0][0]
pycol = np.where(sink_cols == "py")[0][0]
pzcol = np.where(sink_cols == "pz")[0][0]
vxcol = np.where(sink_cols == "vx")[0][0]
vycol = np.where(sink_cols == "vy")[0][0]
vzcol = np.where(sink_cols == "vz")[0][0]
hcol = np.where(sink_cols == "h")[0][0]
mcol = np.where(sink_cols == "m")[0][0]
mtotcol = np.where(sink_cols == "mtot")[0][0]
scol = np.where(sink_cols == "sys_id")[0][0]


df = pd.read_hdf("binary_data.h5", key="data")
unique_binaries = df.index.get_level_values("binary").unique()
with open("dat_stacked.p", "rb") as ff:
    my_data = pickle.load(ff)

with open("lookup_dict_stacked.p", "rb") as ff:
    lookup_dict = pickle.load(ff)

with open("path_lookup_stacked.p", "rb") as ff:
    path_lookup = pickle.load(ff)

ex_time_max = np.load("pmult_before_bin_1.0.npz")["ex_time_max"]
ex_time_max_end = np.load("pmult_before_bin_1.0.npz")["ex_time_max_end"]
ex_filt = ~np.isinf(ex_time_max)
ex_filt = ex_filt & my_data["quasi_filter"]
ex_index = np.where(ex_filt)[0]
# ex_filt = ~np.load("pmult_before_bin_1.0.npz")["pmult_filt"]
# ex_filt = ex_filt & my_data["quasi_filter"]
# ex_index = np.where(ex_filt)[0]
#################################################################################
def rec_sort(my_list):
    my_list_ = copy.deepcopy(my_list)
    if isinstance(my_list_, (int, np.integer)):
        return my_list_

    p1 = my_list_.pop()
    p2 = my_list_.pop()

    if isinstance(p1, (int, np.integer)) and isinstance(p2, (int, np.integer)):
        return [min(p1, p2), max(p1, p2)]
    else:
        return [rec_sort(p2), rec_sort(p1)]


def get_com_winf(paths):
    paths2 = np.copy(paths)
    paths2[np.isinf(paths2)] = 0

    tmp_ms = paths2[:, :, mcol]
    tmp_ms.shape = (len(tmp_ms), -1, 1)
    com = path_divide_3d(np.sum((tmp_ms * paths2[:, :, pxcol:pzcol + 1]), axis=0), np.sum(tmp_ms, axis=0))

    return com

def path_divide_3d(p1, p2):
    assert len(p1) == len(p2)
    diff = np.ones((len(p1), 3)) * np.inf
    filt = (~np.isinf(p1[:,0])) & (~np.isinf(p2[:,0]))
    print(p1[filt].shape, p2[filt].shape)
    diff[filt] = p1[filt] / p2[filt]

    return diff

def get_com(paths):
    tmp_ms = paths[:, :, mcol]
    tmp_ms.shape = (len(tmp_ms), -1, 1)
    com = path_divide_3d(np.sum((tmp_ms * paths[:, :, pxcol:pzcol + 1]), axis=0), np.sum(tmp_ms, axis=0))

    return com

def max_pairwise_dist(pos_list):
    max_dist = 0
    for ii in range(len(pos_list)):
        for jj in range(ii + 1, len(pos_list)):
            tmp_dist = np.linalg.norm(pos_list[ii] - pos_list[jj])
            if tmp_dist > max_dist:
                max_dist = tmp_dist

    return max_dist


##Many arguments can be combined -- just pass the whole dataframe(!)
def plotly_snapshot(tt, ps, comps, comps_curr_list, start_time, end_time, annotations, size_mode="mtot", com_flag=0,
                    max_sep=10000, max_sep_rel=0, halo_col=True, ex_time_start=np.inf, ex_time_end=np.inf):
    fig = go.Figure()
    size_col = mtotcol if halo_col else mcol
    size_scale = 0.0003

    paths = np.array([path_lookup[str(pp)] for pp in ps])
    paths_T = np.transpose(paths, axes=(1,0,2))

    ##mcol of mtotcol should also affect the center of mass calculation...
    coms = get_com_winf(paths)
    delta = np.zeros((end_time + 1, 3))
    if com_flag == 1:
        delta = np.array([coms[start_time]] * len(delta))
    elif com_flag > 0:
        delta = np.array(coms)

    ##If max_sep_rel is specified then
    pos_list = np.array([paths[tmp, tt, pxcol:pxcol + 2] for tmp in range(len(ps))])
    max_sep_b = max_sep
    if (max_sep_rel > 0) and ~(np.any(np.isnan(pos_list[:,0]))) and ~(np.any(np.isinf(pos_list[:,0]))):
        max_sep_b = max_sep_rel * max_pairwise_dist(pos_list) * unit

    names = [f"Star {sid}" for sid in ps]
    masses = paths_T[tt, :, size_col]
    hover_texts = [f"{name}<br>Mass: {mass:.2f} M☉" for name, mass in zip(names, masses)]
    fig.add_trace(go.Scatter(
        x=(paths_T[tt, :, pxcol] - delta[tt, 0]) * unit,
        y=(paths_T[tt, :, pxcol + 1] - delta[tt, 1]) * unit,
        mode="markers",
        marker=dict(
            symbol="square",
            size=3. * np.log10(paths_T[tt, :, size_col] / size_scale),
            color="black",
            opacity=0.5
        ),
        showlegend=False,
        text=hover_texts,
        hoverinfo="text"
    ))

    # Trails and final positions
    # for pidx in range(len(ps)):
    #     # fig.add_trace(go.Scatter(
    #     #     x=paths_T[:tt+1, pidx, pxcol] - delta[:tt+1, 0],
    #     #     y=paths_T[:tt+1, pidx, pxcol+1] - delta[:tt+1, 1],
    #     #     mode="lines",
    #     #     line=dict(color="black"),
    #     #     name=f"Star {ps[pidx]}",
    #     #     showlegend=False
    #     # ))
    #     fig.add_trace(go.Scatter(
    #         x=[(paths_T[tt, pidx, pxcol] - delta[tt, 0]) * unit],
    #         y=[(paths_T[tt, pidx, pxcol + 1] - delta[tt, 1]) * unit],
    #         mode="markers",
    #         marker=dict(
    #             symbol="square",
    #             size=3. * np.log10(paths_T[tt, pidx, size_col] / size_scale),
    #             color="black",
    #             opacity=0.5
    #         ),
    #         showlegend=False
    #     ))

    # Past/future companions (hollow)
    if len(comps) > 0:
        paths_extra = np.array([path_lookup[str(pp)] for pp in comps])
        paths_extra_T = np.transpose(paths_extra, axes=(1, 0, 2))

        names = [f"Star {sid}" for sid in comps]
        masses = paths_extra_T[tt, :, size_col]
        hover_texts = [f"{name}<br>Mass: {mass:.2f} M☉" for name, mass in zip(names, masses)]

        fig.add_trace(go.Scatter(
            x=(paths_extra_T[tt, :, pxcol] - delta[tt, 0]) * unit,
            y=(paths_extra_T[tt, :, pxcol+1] - delta[tt, 1]) * unit,
            mode="markers",
            marker=dict(
                size=3. * np.log10(paths_extra_T[tt, :, size_col] / size_scale),
                color="black",
                opacity=0.5,
                symbol="circle-open"
            ),
            text=hover_texts,
            hoverinfo="text",
            showlegend=False
        ))

    # Current companions (filled edge)
    if len(comps_curr_list) > 0:
        print(comps_curr_list)
        paths_curr = np.array([path_lookup[str(pp)] for pp in comps_curr_list])
        paths_curr_T = np.transpose(paths_curr, axes=(1, 0, 2))

        names = [f"Star {sid}" for sid in comps_curr_list]
        masses = paths_curr_T[tt, :, size_col]
        hover_texts = [f"{name}<br>Mass: {mass:.2f} M☉" for name, mass in zip(names, masses)]

        fig.add_trace(go.Scatter(
            x=(paths_curr_T[tt, :, pxcol] - delta[tt, 0]) * unit,
            y=(paths_curr_T[tt, :, pxcol+1] - delta[tt, 1]) * unit,
            mode="markers",
            marker=dict(
                size=3. * np.log10(paths_curr_T[tt, :, size_col] / size_scale),
                color="red",
                line=dict(color="red", width=1)
            ),
            text=hover_texts,
            hoverinfo="text",
            showlegend=False
        ))

    xcenter = (coms[tt, 0] - delta[tt, 0]) * unit
    ycenter = (coms[tt, 1] - delta[tt, 1]) * unit
    # Axes and annotation
    col_axis = "black"
    if (tt >= ex_time_start) & (tt <= ex_time_end):
        col_axis = "red"
    fig.update_layout(
        xaxis=dict(
            title="x [au]",
            range=[(xcenter - max_sep), (xcenter + max_sep)],
            color=col_axis
        ),
        yaxis=dict(
            title="y [au]",
            range=[(ycenter - max_sep), (ycenter + max_sep)],
            color=col_axis
        ),
        title=annotations[tt],
        height=600,
        width=600
    )
    fig2 = go.Figure(fig)
    fig2.update_layout(
        xaxis=dict(
            title="x [au]",
            range=[(xcenter - max_sep_b), (xcenter + max_sep_b)],
        ),
        yaxis=dict(
            title="y [au]",
            range=[(ycenter - max_sep_b), (ycenter + max_sep_b)],
        ),
        height=600,
        width=600
    )

    return fig, fig2

##Many arguments can be combined -- just pass the dataframe...
def movie(ps, tt, comps, comps_curr, annotations, first_star_snap, tmp_end_snap, max_sep, max_sep_rel, halo_col, ex_time_start, ex_time_end):
    ################# Above do not update to save time!!!##########################################################
    # if case_label is None:
    #     case_label = tmp_bin_idx
    fig, fig2 = plotly_snapshot(tt, ps, comps, comps_curr, first_star_snap, tmp_end_snap, annotations=annotations, com_flag=2,
                          max_sep=max_sep, max_sep_rel=max_sep_rel, halo_col=halo_col, ex_time_start=ex_time_start, ex_time_end=ex_time_end)
    return fig, fig2


# === Initialize app ===
app = dash.Dash(__name__)

# === Layout ===
app.layout = html.Div([
    html.H2("Binary Snapshot Viewer"),
    html.Div(
        [dbc.Label("Binary index"), dcc.Input(id="bin-input",
        type="number",
        min=0,
        max=len(ex_index) - 1,
        value=0)
        ]
    ),##Have option to have to plot range scaled to the binary separation!!!
    html.Div(
        [dbc.Label("(Left plot) Set scale"), dcc.Input(id="max-sep",
                   type="number",
                   value=10000)
         ]
    ),
    html.Div(
        [dbc.Label("(Right plot) Set scale relative to binary separation:"), dcc.Input(id="max-sep-rel",
                   type="number",
                   value=5)
         ]
    ),
    html.Div([
            dbc.Label("Include halo?"),
            dbc.RadioItems(
                id='halo_toggle',
                options=[
                    {'label': 'Include halo', 'value': True},
                    {'label': 'Exclude halo', 'value': False}
                ],
                value=True,
                inline=True
            )
        ], className="mb-1"),
    html.Div([dbc.Label("Time:"),
    html.Button("←", id="step-back", n_clicks=0),
    dcc.Input(
        id="time-input",
        type="number",
        min=0,
        max=490,
        value=422,
        step=1,
        style={"width": "80px"}
    ),
    html.Button("→", id="step-forward", n_clicks=0),
    ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
    html.Div([
        dcc.Graph(id="snapshot-graph"),
        dcc.Graph(id="snapshot-graph2")
    ],  style={'display': 'grid','gridTemplateColumns': '1fr 1fr','gap': '20px'}
    )
])

##Are n_back/n_forward really necessary here??
##Make sure we get only indices that we actually want here e.g. the excahnges(!)
@app.callback(
    Output("snapshot-graph", "figure"),
    Output("snapshot-graph2", "figure"),
    Output("time-input", "value"),
    Input("step-back", "n_clicks"),
    Input("step-forward", "n_clicks"),
    Input("bin-input", "value"),
    Input("time-input", "value"),
    Input("max-sep", "value"),
    Input("max-sep-rel", "value"),
    Input("halo_toggle", "value"),
    State("time-input", "value"),
)
def update_figure(n_back, n_forward, bin_input, input_value, max_sep, max_sep_rel, halo_toggle, current_value):
    ##Only looking at subset of exchange binaries for now
    ##TO DO: Develop ability to look at all binaries.
    tmp_bin_idx = ex_index[bin_input]
    my_bin = df.loc[unique_binaries[tmp_bin_idx]]
    print(my_bin)
    ps = my_bin.index.to_list()
    tmp_end_snap = int(my_data["final_bound_snaps"][tmp_bin_idx]) + 2
    tmp_end_snap = min(int(lookup_dict[ps[0]][0, -1]), tmp_end_snap)

    ##Can we avoid splitting the data up???
    c1 = my_bin.iloc[0]["comps"]
    c2 = my_bin.iloc[1]["comps"]
    times1 = my_bin.iloc[0]["times"]
    times2 = my_bin.iloc[1]["times"]
    hiers1 = my_bin.iloc[0]["hiers"]
    hiers2 = my_bin.iloc[1]["hiers"]
    first_star_snap = int(min(min(times1), min(times2)))

    ##Get only quasi-persistent companions(!!!)
    comps = np.concatenate((np.unique(np.concatenate(c1)), np.unique(np.concatenate(c2))))
    comps = comps[(comps != ps[0]) & (comps != ps[1])]

    annotations1 = [""] * 490
    for ii, ttt in enumerate(times1):
        annotations1[int(ttt)] = rec_sort(hiers1[ii])

    annotations2 = [""] * 490
    for ii, ttt in enumerate(times2):
        annotations2[int(ttt)] = rec_sort(hiers2[ii])

    annotations = [f"{ps[0]} {ps[1]}" + "<br>" + f"{annotations1[ii]}"+ "<br>" + f"{annotations2[ii]}" for ii in range(len(annotations1))]

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "step-back":
        new_t = max(current_value - 1, first_star_snap)
    elif triggered_id == "step-forward":
        new_t = min(current_value + 1, tmp_end_snap)
    elif triggered_id == "bin-input":
        new_t = max(int(ex_time_max[tmp_bin_idx]) - 10, my_data["fst"][tmp_bin_idx])
        # new_t = max(int(my_data["init_bound_snaps"][tmp_bin_idx]) - 10, my_data["fst"][tmp_bin_idx])
    else:
        new_t = current_value

    comps_curr = []
    t_group = (times1, times2)
    for ii, cc in enumerate((c1, c2)):
        tmp_comps_curr = np.array(cc, dtype=object)[np.array(t_group[ii]) == new_t]
        if len(tmp_comps_curr) > 0:
            comps_curr.append(tmp_comps_curr[0])

    if len(comps_curr) > 0:
        comps_curr = np.concatenate(comps_curr)
        comps_curr = np.unique(comps_curr[(comps_curr!=ps[0]) & (comps_curr!=ps[1])])

    ##Refactor -- too many arguments...
    ##Need to pass the time interval over which exchange occurs...
    fig, fig2 = movie(ps, new_t, comps, comps_curr, annotations, first_star_snap, tmp_end_snap, float(max_sep), float(max_sep_rel), halo_toggle,
                      ex_time_max[tmp_bin_idx], ex_time_max_end[tmp_bin_idx])
    return fig, fig2, new_t


if __name__=="__main__":
    app.run(debug=True)
    # df = pd.read_hdf("binary_data.h5", key="data")
    # ##All companions plotted together with hallow symbols
    #
    #
    # movie(df, 72, 300)