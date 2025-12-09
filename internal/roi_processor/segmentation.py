def fill_vertical_gaps(zones, img_width, img_height):
    zones = [z.copy() for z in zones]

    # сортируем по x1
    zones_sorted = sorted(zones, key=lambda z: z["bbox"][0])

    for i in range(len(zones_sorted)):
        for j in range(i + 1, len(zones_sorted)):
            z1 = zones_sorted[i]["bbox"]
            z2 = zones_sorted[j]["bbox"]

            # Проверяем, пересекаются ли они по X
            if z1[2] < z2[0] or z2[2] < z1[0]:
                continue

            # Вертикальный gap
            if z1[3] < z2[1]:
                gap_top = z1[3]
                gap_bottom = z2[1]

                width1 = z1[2] - z1[0]
                width2 = z2[2] - z2[0]

                if width1 >= width2:
                    z1[3] = z2[1]  # расширяем вниз
                else:
                    z2[1] = z1[3]  # расширяем вверх

            elif z2[3] < z1[1]:
                gap_top = z2[3]
                gap_bottom = z1[1]

                width1 = z1[2] - z1[0]
                width2 = z2[2] - z2[0]

                if width1 >= width2:
                    z1[1] = z2[3]  # расширяем вверх
                else:
                    z2[3] = z1[1]  # расширяем вниз

    return zones_sorted

def fill_horizontal_gaps(zones):
    zones = [z.copy() for z in zones]

    zones_sorted = sorted(zones, key=lambda z: z["bbox"][1])

    for i in range(len(zones_sorted)):
        for j in range(i + 1, len(zones_sorted)):
            z1 = zones_sorted[i]["bbox"]
            z2 = zones_sorted[j]["bbox"]

            # Проверяем пересечение по Y
            if z1[3] < z2[1] or z2[3] < z1[1]:
                continue

            # Горизонтальный gap
            if z1[2] < z2[0]:
                h1 = z1[3] - z1[1]
                h2 = z2[3] - z2[1]

                if h1 >= h2:
                    z1[2] = z2[0]  # расширяем вправо
                else:
                    z2[0] = z1[2]  # расширяем влево

            elif z2[2] < z1[0]:
                h1 = z1[3] - z1[1]
                h2 = z2[3] - z2[1]

                if h1 >= h2:
                    z1[0] = z2[2]  # расширяем влево
                else:
                    z2[2] = z1[0]  # расширяем вправо

    return zones_sorted