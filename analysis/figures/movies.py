from starforge_mult_search.analysis.figures.figure_preamble import *
from bash_command import bash_command as bc
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.io as pio

import dash
from dash import dcc, html, Output, Input

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



def plotly_snapshot(tt, ps, comps, comps_curr_list, start_time, end_time, annotations, size_mode="mtot", com_flag=0):
    fig = go.Figure()
    size_col = mtotcol if size_mode == "mtot" else mcol
    size_scale = 0.4

    paths = np.array([path_lookup[str(pp)] for pp in ps])
    paths_T = np.transpose(paths, axes=(1,0,2))
    max_sep = 0.04

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
        fig.add_trace(go.Scatter(
            x=paths_T[:tt+1, pidx, pxcol] - delta[:tt+1, 0],
            y=paths_T[:tt+1, pidx, pxcol+1] - delta[:tt+1, 1],
            mode="lines",
            line=dict(color="black"),
            name=f"Star {ps[pidx]}",
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[paths_T[tt, pidx, pxcol] - delta[tt, 0]],
            y=[paths_T[tt, pidx, pxcol+1] - delta[tt, 1]],
            mode="markers",
            marker=dict(
                symbol="square",
                size=paths_T[tt, pidx, size_col] / size_scale,
                color="black"
            ),
            showlegend=False
        ))

    # Past/future companions (hollow)
    if len(comps) > 0:
        paths_extra = np.array([path_lookup[str(pp)] for pp in comps])
        paths_extra_T = np.transpose(paths_extra, axes=(1, 0, 2))
        fig.add_trace(go.Scatter(
            x=paths_extra_T[tt, :, pxcol] - delta[tt, 0],
            y=paths_extra_T[tt, :, pxcol+1] - delta[tt, 1],
            mode="markers",
            marker=dict(
                size=paths_extra_T[tt, :, size_col] / size_scale,
                color=colorblind_palette[0],
                opacity=0.5,
                symbol="circle-open"
            ),
            showlegend=False
        ))

    # Current companions (filled edge)
    if len(comps_curr_list) > 0:
        paths_curr = np.array([path_lookup[str(pp)] for pp in comps_curr_list])
        paths_curr_T = np.transpose(paths_curr, axes=(1, 0, 2))
        fig.add_trace(go.Scatter(
            x=paths_curr_T[tt, :, pxcol] - delta[tt, 0],
            y=paths_curr_T[tt, :, pxcol+1] - delta[tt, 1],
            mode="markers",
            marker=dict(
                size=paths_curr_T[tt, :, size_col] / size_scale,
                color=colorblind_palette[0],
                line=dict(color=colorblind_palette[0], width=1)
            ),
            showlegend=False
        ))

    # Axes and annotation
    fig.update_layout(
        xaxis=dict(
            title="x [pc]",
            range=[xcenter - 10*max_sep, xcenter + 10*max_sep]
        ),
        yaxis=dict(
            title="y [pc]",
            range=[ycenter - 10*max_sep, ycenter + 10*max_sep]
        ),
        title=annotations[tt]
    )
    return fig

def movie(df, tmp_bin_idx, tt, case_label=None):
    unique_binaries = df.index.get_level_values("binary").unique()
    my_bin = df.loc[unique_binaries[tmp_bin_idx]]
    print(my_bin)
    ps = my_bin.index.to_list()
    tmp_end_snap = int(lookup_dict[ps[0]][0, -1])

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
        tmp_comps_curr = np.array(cc, dtype=object)[np.array(t_group[ii]) == tt]
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
    if case_label is None:
        case_label = tmp_bin_idx
    fig = plotly_snapshot(tt, ps, comps, comps_curr, first_star_snap, tmp_end_snap, annotations=annotations, com_flag=0)
    return fig


# === Initialize app ===
app = dash.Dash(__name__)

# === Layout ===
app.layout = html.Div([
    html.H2("Binary Snapshot Viewer"),
    dcc.Slider(
        id="time-slider",
        min=0,
        max=489,
        step=1,
        value=300,
        tooltip={"placement": "bottom", "always_visible": True},
    ),
    dcc.Graph(id="snapshot-graph")
])

@app.callback(
    Output("snapshot-graph", "figure"),
    Input("time-slider", "value")
)
def update_figure(tt):
    df = pd.read_hdf("binary_data.h5", key="data")
    fig = movie(df, 72, tt)
    return fig


if __name__=="__main__":
    app.run(debug=True)
    # df = pd.read_hdf("binary_data.h5", key="data")
    # ##All companions plotted together with hallow symbols
    #
    #
    # movie(df, 72, 300)