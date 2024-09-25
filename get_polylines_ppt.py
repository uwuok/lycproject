import cv2
import numpy as np

# 讀取圖片
image = cv2.imread('new.png')  # 請將 'new2.png' 替換為你的圖片檔名

# 定義多邊形的座標點，(x, y) 格式，並以順時針方向排列
# ROI 後 左上 ps2 new
# points = np.array([[2, 4], [922, 36],
#                    [778, 116], [690, 184],
#                    [550, 304], [478, 372],
#                    [414, 448], [358, 524],
#                    [318, 584], [282, 648],
#                    [230, 748], [198, 820],
#                    [154, 932], [126, 1028],
#                    [110, 1100], [106, 1116],
#                    [66, 1116], [2, 436]])

# ROI 後左上 ps2 old (ppt)
# points = np.array([[0, 0], [538, 0], [894, 74],
#                    [798, 134], [766, 156], [698, 202],
#                    [622, 266], [578, 310], [522, 362],
#                    [426, 478], [374, 550], [318, 626],
#                    [294, 674], [254, 758], [210, 842],
#                    [182, 910], [158, 982], [134, 1078],
#                    [130, 1114], [22, 1114], [0, 974]
#                    ])

# ROI 後左下
# points = np.array([[7,38],[69,207],[146,376],
#                    [176,438],[253,546],[330,653],[376,730],[438,807],[484,869],[546,930],[607,992],[684,1053],
# [761,1130],[823,1176],[884,1223],
# [992,1315],[1053,1346],[1130,1392],
# [1192,1423],[1238,1453],[1238,1576],[1269,1807],[1300,2084],[1315,2207],[1346,2376],[1469,3238],[346,3469]
# ])


# 大圖 ps1 (左下)
# points = np.array([[560, 2504], [2054, 2504], [2054, 5983], [560, 5983]])

# 大圖 ps2 (左上)

points = np.array([[370, 410], [1342, 410], [1342, 1529], [370, 1529]])


# 繪製多邊形
cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=15)

# 標註座標點
for i, point in enumerate(points):
    # (x, y)
    text = f'({point[0]}, {point[1]})'

    # 放置標註文字，調整座標位置避免重疊
    # cv2.putText(image, text, (point[0] + 10, point[1] - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, text, (point[0], point[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 8, cv2.LINE_AA)


# 儲存繪製結果
cv2.imwrite('ptt.png', image)  # 將結果儲存為 'ptt.png'