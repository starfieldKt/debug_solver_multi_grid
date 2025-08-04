import math
import numpy as np
import iric
import sys
import os

print("----------Start----------")

###############################################################################
# CGNSを開く
###############################################################################

# iRICで動かす時用
# =============================================================================
if len(sys.argv) < 2:
    print("Error: CGNS file name not specified.")
    exit()

cgns_name = sys.argv[1]

print("CGNS file name: " + cgns_name)

# CGNSをオープン
fid = iric.cg_iRIC_Open(cgns_name, iric.IRIC_MODE_MODIFY)

# コマンドラインで動かす時用
# =============================================================================

# CGNSをオープン
# fid = iric.cg_iRIC_Open("./project/Case1.cgn", iric.IRIC_MODE_MODIFY)

# 分割保存したい場合はこれを有効にする
# os.environ['IRIC_SEPARATE_OUTPUT'] = '1'

###############################################################################
# 古い計算結果を削除
###############################################################################

iric.cg_iRIC_Clear_Sol(fid)

###############################################################################
# 計算条件を読み込み
###############################################################################

# 格子サイズを読み込み
isize, jsize = iric.cg_iRIC_Read_Grid2d_Str_Size(fid)

# 格子点の座標読み込み
# --------------------------------------------------
# メモ
# --------------------------------------------------
# CGNSから読み込む時は1次元配列、順番は以下
# --------------------------------------------------
#      j
#      ↑
#     4| 24, 25, 26, 27, 28, 29
#     3| 18, 19, 20, 21, 22, 23
#     2| 12, 13, 14, 15, 16, 17
#     1|  6,  7,  8,  9, 10, 11
#     0|  0,  1,  2,  3,  4,  5
#       ----------------------- →　i
#         0   1   2   3   4   5
# --------------------------------------------------
grid_x_arr_2d, grid_y_arr_2d = iric.cg_iRIC_Read_Grid2d_Coords(fid)
grid_x_arr_2d = grid_x_arr_2d.reshape(jsize, isize).T
grid_y_arr_2d = grid_y_arr_2d.reshape(jsize, isize).T

# 計算時間を読み込み
time_end = iric.cg_iRIC_Read_Integer(fid, "time_end")

# 読み込んだ格子サイズをコンソールに出力
print("Grid size:")
print("    isize= " + str(isize))
print("    jsize= " + str(jsize))

# gridID_2d = iric.cg_iRIC_Write_Grid2d_Coords_WithGridId(fid,isize,jsize,grid_x_arr_2d.flatten(order="F"),grid_y_arr_2d.flatten(order="F"))

###############################################################################
# 3Dの格子をつくる
###############################################################################

# z方向格子数
ksize = iric.cg_iRIC_Read_Integer(fid, "ksize")
# 平均水位
average_WL = iric.cg_iRIC_Read_Real(fid, "average_WL")
# 波高
wave_height = iric.cg_iRIC_Read_Real(fid, "wave_height")

# x,yは2次元格子点座標をk方向にコピーして作成
grid_x_arr_3d =grid_x_arr_2d[:, :, np.newaxis] * np.ones(ksize)
grid_y_arr_3d =grid_y_arr_2d[:, :, np.newaxis] * np.ones(ksize)

print("----------mainloop start----------")

###############################################################################
# メインループスタート
###############################################################################

for t in range(time_end + 1):

    ###############################################################################
    # 水深の計算
    ###############################################################################
    # 水深は平均水位に対して波高wave_heightのsinとcosの2次元波を加えたものとする
    # i方向にはsin波、j方向にはcos波を加える
    # また、tによる位相差をつける
    
    
    # 2次元格子点の水深を計算
    # i方向の倍率
    i_indices = np.arange(isize)
    i_ratios = np.sin(np.pi * 2 * (i_indices / isize + t/60))

    # j方向の倍率
    j_indices = np.arange(jsize)
    j_ratios = np.cos(np.pi * 2 * (j_indices / jsize + t/30))

    # i_ratios と j_ratios の外積を計算して水深を計算
    depth_2d = average_WL + wave_height * np.outer(i_ratios, j_ratios)

    # 3次元格子点のz座標を計算
    # depth_2dが水深なので、z座標は水深に対して等間隔に配置する
    # k=ksize-1のz座標がdepth_2dになるように配置する
    grid_z_arr_3d = np.linspace(0, depth_2d[:, :, np.newaxis], ksize, axis=2)

    # 相対Z位置を生成（0〜1に正規化されたk方向）
    relative_z = np.linspace(0, 1, ksize)
    relative_z_3d = np.tile(relative_z, (isize, jsize, 1))

    # 初回のみ、3次元格子をCGNSに書き込む
    if t == 0:
        gid_3d = iric.cg_iRIC_Write_Grid3d_Coords_WithGridId(fid,isize,jsize,ksize,grid_x_arr_3d.flatten(order="F"),grid_y_arr_3d.flatten(order="F"),grid_z_arr_3d.flatten(order="F"))

    ###########################################################################
    # 結果の書き込みスタート
    ###########################################################################

    # 時間ごとの書き込み開始をGUIに伝える
    iric.cg_iRIC_Write_Sol_Start(fid)

    # 時刻を書き込み
    iric.cg_iRIC_Write_Sol_Time(fid, float(t))

    # 2次元格子に対する書き込み
    # 水深を書き込み（格子点実数値）
    iric.cg_iRIC_Write_Sol_Node_Real_WithGridId(fid,1, "depth_2d", depth_2d.flatten(order="F"))

    # 3次元格子に対する書き込み
    # 水深を書き込み（格子点実数値）
    iric.cg_iRIC_Write_Sol_Grid3d_Coords_WithGridId(fid, gid_3d, grid_x_arr_3d.flatten(order="F"), grid_y_arr_3d.flatten(order="F"), grid_z_arr_3d.flatten(order="F"))

    iric.cg_iRIC_Write_Sol_Node_Real_WithGridId(fid,gid_3d, "Sigma", relative_z_3d.flatten(order="F"))

    # CGNSへの書き込み終了をGUIに伝える
    iric.cg_iRIC_Write_Sol_End(fid)

    # コンソールに時間を出力
    print("t= " + str(t))

    # 計算結果の再読み込みが要求されていれば出力を行う
    iric.cg_iRIC_Check_Update(fid)

    # 計算のキャンセルが押されていればループを抜け出して出力を終了する。
    canceled = iric.iRIC_Check_Cancel()
    if canceled == 1:
        print("Cancel button was pressed. Calculation is finishing. . .")
        break

print("----------finish----------")

###############################################################################
# 計算終了処理
###############################################################################
iric.cg_iRIC_Close(fid)

