import numpy as np
import pandas as pd
from lxml import etree
from pykml.factory import KML_ElementMaker as KML
import pymap3d as pm
import os

'''
 mainDir - Dir where train/test data is
 data2WorkOn - Name of the data CSV file
 class2show - Which class in CSV file to draw, or (-1) if using test set or lines2show parameter
 lines2show - Which lines in CSV file to draw
 nedOrigin - Geo coordinates where XYZ = 0
 openFileWhenDone - Open KML after it is ready (KML files have to be associated with Google Earth)
'''

mainDir = os.path.expanduser('~/hack/rafael/')
data2WorkOn = 'test_sample'
# data2WorkOn = 'train_sample'
class2show = -1  # for 'train' only
lines2show = range(0, 300)  # for 'test' (or 'train if class2show == -1)
nedOrigin = (31.784491, 35.214245, 0)
openFileWhenDone = True


def placemark_line(name, lla_str, clr_str):
    return KML.Placemark(
            KML.name(name),
            KML.Style(
                KML.LineStyle(
                    KML.color(clr_str),  # "ff00FF40") ,
                    KML.width("1")
                )
            ),
            KML.LineString(
                KML.altitudeMode("absolute"),
                KML.coordinates(
                    lla_str
                )
            )
        )


def placemark_points(name, lla_str, clr_str):
    points = [KML.Point(KML.altitudeMode("absolute"),
                        KML.coordinates(x)) for x in lla_str.split()]
    g = KML.MultiGeometry()
    for p in points:
        g.append(p)
    return KML.Placemark(
            KML.name(''),
            KML.Style(
                KML.LineStyle(
                    KML.color("#ff0000"),  # "ff00FF40") ,
                    KML.width("4")
                )
            ),
            g
        )


def ned2geodetic2String(csvSet, line):
    lla_str = ''
    for iii in range(0, 15):
        # print csvSet.iloc[line]
        northCoor = csvSet.iloc[line]['posX_' + str(iii)]
        eastCoor = csvSet.iloc[line]['posY_' + str(iii)]
        downCoor = -csvSet.iloc[line]['posZ_' + str(iii)]
        if np.isnan(northCoor):
            continue
        lla = pm.ned2geodetic(northCoor, eastCoor, downCoor, nedOrigin[0], nedOrigin[1], nedOrigin[2])
        lla_str = lla_str + str(lla[1]) + "," + str(lla[0]) + "," + str(lla[2]) + " "
    return lla_str


name = 'Rocket_Data_Science_test'
if class2show > 0:
    name = name + '_' + str(class2show)
doc = KML.kml(
    KML.Document(
        KML.name(name),
        KML.Folder(
            KML.name('UP')
        ),
        KML.Folder(
            KML.name('DOWN')
        )
    )
)

print('Loading...')
csvSet = pd.read_csv(mainDir + data2WorkOn + '.csv')

print('Working...')
if data2WorkOn.find('train') and class2show > 0:
    csvSubSet = csvSet[csvSet.iloc[:]['class'] == class2show]
    lines2show = csvSubSet.index

for line in lines2show:
    if line >= len(csvSet):
        continue
    lla_str = ned2geodetic2String(csvSet, line)
    velColor = hex(int(round(np.linalg.norm((csvSet.iloc[line]['velX_0'],
                                             csvSet.iloc[line]['velY_0'],
                                             csvSet.iloc[line]['velZ_0'])) / 1500 * 255)))[-2:]
    if csvSet.iloc[line]['velZ_0'] > 0:
        clr_str = "FF" + velColor + "0000"
        folderName = 0
    else:
        clr_str = "FF0000" + velColor
        folderName = 1
    doc.Document.Folder[folderName].append(
        placemark_line('line_' + str(line), lla_str, clr_str)
    )
    doc.Document.Folder[folderName].append(
        placemark_points('multipoint_' + str(line), lla_str, clr_str)
    )


print('Saving...')
# print etree.tostring(doc, pretty_print=True)

fname = mainDir + name + '.kml'
with open(fname, 'wb') as outfile:
    outfile.write(etree.tostring(doc, pretty_print=True))

os.chmod(fname, 0o766)

print('Done!')
# if openFileWhenDone:
#     os.system(fname)
