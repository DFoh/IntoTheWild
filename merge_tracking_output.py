import json, pandas as pd
from pathlib import Path

def bbox_to_uv(x1,y1,x2,y2, mode="center"):
    if mode=="center": return ((x1+x2)/2.0, (y1+y2)/2.0)
    if mode=="bottom_center": return ((x1+x2)/2.0, max(y1,y2))
    raise ValueError("mode")

def merge_to_jsonl(csv_paths, out_jsonl, uv_mode="center", top1=False):
    dfs=[]
    for p in map(Path, csv_paths):
        cam=p.stem
        df=pd.read_csv(p)
        uv=[bbox_to_uv(*r[3:7], uv_mode) for r in df.itertuples(index=False)]
        df=df.assign(u=[u for u,_ in uv], v=[v for _,v in uv], cam=cam)
        dfs.append(df[["frame","cam","u","v","x1","y1","x2","y2","conf","cls"]])
    all_df=pd.concat(dfs, ignore_index=True)

    # keep only the cam id in the cam column
    all_df["cam"]=all_df["cam"].apply(lambda x: x.split("_")[-2])

    with open(out_jsonl,"w") as f:
        for frame, g in all_df.groupby("frame", sort=True):
            points={}
            for cam, gc in g.groupby("cam"):
                gc=gc.sort_values("conf", ascending=False)
                if top1: gc=gc.head(1)
                items=[]
                for det_id, r in enumerate(gc.itertuples(index=False)):
                    items.append({
                        "det_id": det_id,
                        "uv":[float(r.u), float(r.v)],
                        "bbox":[float(r.x1),float(r.y1),float(r.x2),float(r.y2)],
                        "conf":float(r.conf),
                        "cls":int(r.cls)
                    })
                points[cam]=items
            f.write(json.dumps({"frame": int(frame), "points": points})+"\n")


if __name__ == '__main__':

    # csv_path_1 = "output_tracks/Running trial Markerless 4_Miqus_12_26075_tracks.csv"
    # csv_path_2 = "output_tracks/Running trial Markerless 4_Miqus_3_26071_tracks.csv"
    # csv_path_3 = "output_tracks/Running trial Markerless 4_Miqus_5_26153_tracks.csv"
    # csv_path_4 = "output_tracks/Running trial Markerless 4_Miqus_13_26078_tracks.csv"
    # csv_paths = [csv_path_1, csv_path_2, csv_path_3, csv_path_4]

    csv_paths = list(Path("output_tracks").glob("Running trial*.csv"))


    out_path = "output_tracks/merged_tracking_output.jsonl"

    merge_to_jsonl(csv_paths, out_path, uv_mode="center")
