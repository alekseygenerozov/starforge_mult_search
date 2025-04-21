from starforge_mult_search.analysis.figures.figure_preamble import *
from bash_command import bash_command as bc
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.io as pio

import dash
from dash import dcc, html, Output, Input, State

##We can store data in a better organized way(!!!)
df = pd.read_hdf("binary_data.h5", key="data")
unique_binaries = df.index.get_level_values("binary").unique()


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


##Many arguments can be combined -- just pass the whole dataframe(!)
def plotly_snapshot(tt, ps, comps, comps_curr_list, start_time, end_time, annotations, size_mode="mtot", com_flag=0,
                    max_sep=0.04):
    fig = go.Figure()
    size_col = mtotcol if size_mode == "mtot" else mcol
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
    xcenter = coms[tt, 0] - delta[tt, 0]
    ycenter = coms[tt, 1] - delta[tt, 1]

    # Trails and final positions
    for pidx in range(len(ps)):
        # fig.add_trace(go.Scatter(
        #     x=paths_T[:tt+1, pidx, pxcol] - delta[:tt+1, 0],
        #     y=paths_T[:tt+1, pidx, pxcol+1] - delta[:tt+1, 1],
        #     mode="lines",
        #     line=dict(color="black"),
        #     name=f"Star {ps[pidx]}",
        #     showlegend=False
        # ))
        fig.add_trace(go.Scatter(
            x=[paths_T[tt, pidx, pxcol] - delta[tt, 0]],
            y=[paths_T[tt, pidx, pxcol+1] - delta[tt, 1]],
            mode="markers",
            marker=dict(
                symbol="square",
                size=3. * np.log10(paths_T[tt, pidx, size_col] / size_scale),
                color="black"
            ),
            showlegend=False
        ))

    # Past/future companions (hollow)
    if len(comps) > 0:
        paths_extra = np.array([path_lookup[str(pp)] for pp in comps])
        paths_extra_T = np.transpose(paths_extra, axes=(1, 0, 2))

        names = [f"Star {sid}" for sid in comps]
        masses = paths_extra_T[tt, :, size_col]
        hover_texts = [f"{name}<br>Mass: {mass:.2f} M☉" for name, mass in zip(names, masses)]

        fig.add_trace(go.Scatter(
            x=paths_extra_T[tt, :, pxcol] - delta[tt, 0],
            y=paths_extra_T[tt, :, pxcol+1] - delta[tt, 1],
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
            x=paths_curr_T[tt, :, pxcol] - delta[tt, 0],
            y=paths_curr_T[tt, :, pxcol+1] - delta[tt, 1],
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

    # Axes and annotation
    fig.update_layout(
        xaxis=dict(
            title="x [pc]",
            range=[xcenter - 10*max_sep, xcenter + 10*max_sep],
        ),
        yaxis=dict(
            title="y [pc]",
            range=[ycenter - 10*max_sep, ycenter + 10*max_sep],
        ),
        title=annotations[tt],
        height=600,
        width=600
    )
    return fig

##Many arguments can be combined -- just pass the dataframe...
def movie(ps, tt, comps, comps_curr, annotations, first_star_snap, tmp_end_snap, max_sep):
    ################# Above do not update to save time!!!##########################################################
    # if case_label is None:
    #     case_label = tmp_bin_idx
    fig = plotly_snapshot(tt, ps, comps, comps_curr, first_star_snap, tmp_end_snap, annotations=annotations, com_flag=0,
                          max_sep=max_sep)
    return fig


# === Initialize app ===
app = dash.Dash(__name__)

# === Layout ===
app.layout = html.Div([
    html.H2("Binary Snapshot Viewer"),
    html.Div(
        [dcc.Input(id="bin-input",
        type="number",
        min=0,
        max=100,
        value=21)
        ]
    ),
    html.Div(
        [dcc.Input(id="max-sep",
                   type="number",
                   value=0.04)
         ]
    ),
    html.Div([
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
    dcc.Graph(id="snapshot-graph")
])


##Make sure we get only indices that we actually want here e.g. the excahnges(!)
@app.callback(
    Output("snapshot-graph", "figure"),
    Output("time-input", "value"),
    Input("step-back", "n_clicks"),
    Input("step-forward", "n_clicks"),
    Input("bin-input", "value"),
    Input("time-input", "value"),
    Input("max-sep", "value"),
    State("time-input", "value"),
)
def update_figure(n_back, n_forward, bin_input, input_value, max_sep, current_value):
    ##Index of binary -- Could do either (i) Index within the full list of binaries or (ii) Index among binaries of interest.
    tmp_bin_idx = bin_input
    my_bin = df.loc[unique_binaries[tmp_bin_idx]]
    print(my_bin)
    ps = my_bin.index.to_list()
    tmp_end_snap = int(lookup_dict[ps[0]][0, -1])

    ##Can we avoid splitting the data up???
    c1 = my_bin.iloc[0]["comps"]
    c2 = my_bin.iloc[1]["comps"]
    times1 = my_bin.iloc[0]["times"]
    times2 = my_bin.iloc[1]["times"]
    hiers1 = my_bin.iloc[0]["hiers"]
    hiers2 = my_bin.iloc[1]["hiers"]
    first_star_snap = int(min(min(times1), min(times2)))


    comps = np.concatenate((np.unique(np.concatenate(c1)), np.unique(np.concatenate(c2))))
    comps = comps[(comps != ps[0]) & (comps != ps[1])]

    comps_curr = []
    t_group = (times1, times2)
    for ii, cc in enumerate((c1, c2)):
        tmp_comps_curr = np.array(cc, dtype=object)[np.array(t_group[ii]) == input_value]
        if len(tmp_comps_curr) > 0:
            comps_curr.append(tmp_comps_curr[0])

    if len(comps_curr) > 0:
        comps_curr = np.concatenate(comps_curr)
        comps_curr = np.unique(comps_curr[(comps_curr!=ps[0]) & (comps_curr!=ps[1])])


    annotations1 = [""] * 490
    for ii, ttt in enumerate(times1):
        annotations1[int(ttt)] = hiers1[ii]

    annotations2 = [""] * 490
    for ii, ttt in enumerate(times2):
        annotations2[int(ttt)] = hiers2[ii]

    annotations = [f"{annotations1[ii]}\n{annotations2[ii]}" for ii in range(len(annotations1))]

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "step-back":
        new_t = max(current_value - 1, first_star_snap)
    elif triggered_id == "step-forward":
        new_t = min(current_value + 1, tmp_end_snap)
    else:
        new_t = int(my_data["init_bound_snaps"][tmp_bin_idx]) - 10

    fig = movie(ps, new_t, comps, comps_curr, annotations, first_star_snap, tmp_end_snap, float(max_sep))
    return fig, new_t


if __name__=="__main__":
    app.run(debug=True)
    # df = pd.read_hdf("binary_data.h5", key="data")
    # ##All companions plotted together with hallow symbols
    #
    #
    # movie(df, 72, 300)