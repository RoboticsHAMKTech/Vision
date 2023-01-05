import optparse
from builtins import print
import json
import cv2
import sys
import numpy as np
from numpy import dtype
from Functions import *
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import datetime
import zivid
from sample_utils.display import display_depthmap, display_pointcloud, display_rgb
from sample_utils.paths import get_sample_data_path
from zivid.capture_assistant import SuggestSettingsParameters


glob_center_bbox = [] #global variable for center of bounding box
def _convert_2_2d(point_cloud: zivid.PointCloud, file_name: str) -> None:
    """Convert from point cloud to 2D image.
    Args:
        point_cloud: A handle to point cloud in the GPU memory
        file_name: File name without extension
    """
    print(f"Saving the frame to {file_name}")
    rgba = point_cloud.copy_data("rgba")
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(file_name, bgr)


def _flatten_point_cloud(point_cloud: zivid.PointCloud) -> np.ndarray:
    """Convert from point cloud to flattened point cloud (with numpy).
    Args:
        point_cloud: A handle to point cloud in the GPU memory
    Returns:
        A 2D numpy array, with 8 columns and npixels rows
    """
    # Convert to numpy 3D array
    point_cloud = np.dstack([point_cloud.copy_data("xyz"), point_cloud.copy_data("rgba"), point_cloud.copy_data("snr")])
    # Flattening the point cloud
    flattened_point_cloud = point_cloud.reshape(-1, 8)

    # Removing nans
    return flattened_point_cloud[~np.isnan(flattened_point_cloud[:, 0]), :]

def _point_cloud_to_cv_bgr(point_cloud: zivid.PointCloud) -> np.ndarray:
    """Get bgr image from frame.

    Args:
        point_cloud: Zivid point cloud

    Returns:
        bgr: BGR image (HxWx3 ndarray)

    """
    rgba = point_cloud.copy_data("rgba")
    # Applying color map
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    return bgr

def _point_cloud_to_cv_z(point_cloud: zivid.PointCloud) -> np.ndarray:
    """Get depth map from frame.

    Args:
        point_cloud: Zivid point cloud

    Returns:
        depth_map_color_map: Depth map (HxWx1 ndarray)

    """
    depth_map = point_cloud.copy_data("z")
    depth_map_uint8 = ((depth_map - np.nanmin(depth_map)) / (np.nanmax(depth_map) - np.nanmin(depth_map)) * 255).astype(
        np.uint8
    )

    depth_map_color_map = cv2.applyColorMap(depth_map_uint8, cv2.COLORMAP_VIRIDIS)

    # Setting nans to black
    depth_map_color_map[np.isnan(depth_map)[:, :]] = 0

    return depth_map_color_map


class Rcet:
    def __init__(self,left,right,top,bottom):
        if left<0:
            self.left=0
        else:
            self.left=left
        if right>639:
            self.right=639
        else:
            self.right=right
        if top<0:
            self.top=0
        else:
            self.top=top
        if bottom>479:
            self.bottom=479
        else:
            self.bottom=bottom

class Objection:
    def CheckCenterPoint(self):
        self.prePoint=np.array([(self.Area.top+self.Area.bottom)/2,(self.Area.right+self.Area.left)/2])
        self.Point=self.prePoint
    def __init__(self,rcet,classname):
        self.Area=rcet
        self.classname=classname
        self.Enable=False
        self.Object_Area=D435_para.colormat[self.Area.left:self.Area.right,self.Area.top:self.Area.bottom]
        self.CheckCenterPoint()
        self.Transform_ImgtoCam()
    def Transform_ImgtoCam(self):
        New_PT=Position_tansform(self.Point)
        self.PCL=New_PT.Report_PCL()


class Position_tansform:
    def __init__(self,RGB_pix_Position):
        # self.Depth_cam,self.Color_cam=np.mat(np.array([0.0,0.0,0.0]))
        self.RGB_Pix_POS=RGB_pix_Position;
    def Report_PCL(self):
        pix_3d=np.mat(np.append(self.RGB_Pix_POS,[1.00]))
        self.Color_cam=D435_para.color_inner_matirx.I*pix_3d.T #D435_para.depthmat[int(self.RGB_Pix_POS[0]),int(self.RGB_Pix_POS[1])]
        self.Depth_cam=D435_para.color_to_depth_rotation.I*(self.Color_cam-D435_para.color_to_depth_translation.T)
        D_m=D435_para.depth_inner_matrix*self.Depth_cam
        # print(Color3[2,0],self.Color_cam[2,0])
        if D_m[0,0]/D_m[2,0]<479:
            P1=D_m[0,0]/D_m[2,0]
        else:
            P1=479
        if D_m[1,0]/D_m[2,0]<639:
            P2=D_m[1,0]/D_m[2,0]
        else:
            P2=639
        self.Depth_pix=np.array([P1,P2])
        Image_pix=np.mat(np.append(self.Depth_pix,[1.00]))
        PCL=D435_para.depth_inner_matrix.I*Image_pix.T*D435_para.depthmat[int(P1),int(P2)]
        self.PCL=[int(PCL[0,0]),int(PCL[1,0]),int(PCL[2,0])]
        print("PCL COORDS=> ",self.PCL)
        return self.PCL

# config
class Realsense_para:
    def __init__(self,ci,di,ex):
        self.color_inner_matirx=np.mat(np.array([[ci.fx,0,ci.ppx],[0,ci.fy,ci.ppy],[0,0,1]]))
        self.depth_inner_matrix=np.mat(np.array([[di.fx,0,di.ppx],[0,di.fy,di.ppy],[0,0,1]]))
        self.color_to_depth_rotation=np.mat(np.array(ex.rotation).reshape(3,3))##相机转换矩阵 旋转矩阵
        self.color_to_depth_translation=np.mat(np.array(ex.translation))###平移矩阵
    def refresh_mat(self):
        self.frames = pipeline.wait_for_frames()
        self.depth = self.frames.get_depth_frame()
        self.color = self.frames.get_color_frame()
        hole_filling = rs.hole_filling_filter()
        self.depth = hole_filling.process(self.depth)
        # depth_profile = depth.get_profile()
        # color_profile = color.get_profile()
        # print(depth_profile)
        # print(color_profile)
        self.depthmat=np.asanyarray(self.depth.get_data())
        self.colormat=np.asanyarray(self.color.get_data())

### Converting 2D image coordinates to 3D coordinates using ROS + Intel Realsense D435/Kinect #######

def convert_depth_to_phys_coord_using_realsense(x, y, depth):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    profi = profile.get_stream(rs.stream.depth)
    intrinss = profi.as_video_stream_profile().get_intrinsics()
    print("test focal fx= ", intrinss.fx)
    print("test focal fy= ", intrinss.fy)
    print("test ppx  = ", intrinss.ppx)
    print("test ppy  = ", intrinss.ppy)
    #_intrinsics.model = cameraInfo.distortion_model
    result = rs.rs2_deproject_pixel_to_point(intrinss, [x, y], depth)
    #result[0]: right, result[1]: down, result[2]: forward
    return result
    #return result[2], -result[0], -result[1]

##################################################################################################

def get_zivid_coords(height, width, pixels_to_display):
    #b_box = postprocess(timg, qualify_result, outs, tconfThreshold, tnmsThreshold)
    list_of_zivid_coords = []
    b_box = glob_center_bbox
    #print("b_box", b_box)
    #print("type(b_box)", type(b_box))
    for index in range(len(b_box)):
        value = b_box[index]
        print("Tomato: ",index+1, value)
        #print("Point cloud information, index:", index)

        #print(f"Number of points: {value[1] * value[0]}")
        #print(f"Height: {value[1]}, Width: {value[0]}")
        print("Iterating over point cloud and extracting X, Y, Z "f"for central {pixels_to_display} x {pixels_to_display} pixels")
        x_zivid = xyz[value[1], value[0], 0]
        y_zivid = xyz[value[1], value[0], 1]
        z_zivid = xyz[value[1], value[0], 2]
        const = 1
        #print("x_zivid, y_zivid, z_zivid: ", x_zivid, y_zivid, z_zivid)
        list_of_zivid_coords.append([x_zivid, y_zivid, z_zivid, const])
        print("list_of_zivid_coords: ", list_of_zivid_coords)
    return list_of_zivid_coords

def cleanFile(fileName):
    # open file for reading data
    if os.path.exists(fileName):
        with open(fileName, 'w') as f:
            f.write("")

# save coordinates x,y,z of tomato to file
def saveCoordsFile(tomato_coords_tuples, fileName, mode):
    print(tomato_coords_tuples)

    # open file for ADDING data
    with open(fileName, mode) as f:
        for tomato in range(len(tomato_coords_tuples)):
            print(tomato_coords_tuples[tomato], end="")
            s = str(tomato_coords_tuples[tomato])
            s = s.replace(",", "")
            s = s.replace("[", "")
            s = s.replace("]", "")
            # write three elements of tuple to file x, y, z
            f.write("{}".format(s))

            #f.write("{} {} {}".format(tomato_coords_tuples[0], tomato_coords_tuples[1], tomato_coords_tuples[2]))
            f.write('\n')

def saveCoordsFileFromList(tomato_coords_list_of_tuples, fileName, mode):
    print(tomato_coords_list_of_tuples)

    # open file for ADDING data
    with open(fileName, mode) as f:
        for item in tomato_coords_list_of_tuples:
            # write three elements of tuple to file
            f.write("{} {} {} {}".format(item[0], item[1], item[2], item[3]))
            f.write('\n')

def readCoordsFile(fileName):
    # open file for reading data
    # read list of lists for sorting
    listTouplesOfCoords = []
    #arrayCoords = np.array([])
    with open(fileName, 'r') as f:
        for line in f.readlines():         # read rest of lines
            x, y, z, const = line.split()
            arrayCoords = np.append('d',(float(x), float(y), float(z), float(const)))
            print("arrayCoords= ", arrayCoords)
            listTouplesOfCoords.append((float(x), float(y), float(z), float(const)))
            print("list appended: ", listTouplesOfCoords)
    listTouplesOfCoords.sort(key=lambda x: x[2])   # sort by third element of tuple (z) - distance from camera to tomato in mm
    return listTouplesOfCoords

def readForArray(fileName):
    arrayCoords = np.array([[0, 0, 0, 0]])
    with open(fileName, 'r') as f:
        for line in f.readlines():         # read rest of lines
            # read from  item three elements of tuple to file
            x, y, z, const = line.split()
            arrayCoords = np.vstack([arrayCoords, np.array([float(x), float(y), float(z), float(const)])])
    return arrayCoords

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(result_image, classId, conf, left, top, right, bottom):
    if first_pass:
        color = 0, 255, 0
    else:
        color = 255, 255, 255
    cv.rectangle(result_image, (left, top), (right, bottom), color, 3, 3)

def drawBad(result_image, left, top, right, bottom):
    cv.rectangle(result_image, (left, top), (right, bottom), (0, 0, 255), 3, 4)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(color_image, result_image, outs, confThreshold, nmsThreshold):
    frameHeight = color_image.shape[0]
    frameWidth = color_image.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    #center_bounding_box = []
    global objVector
    for out in outs:
        #print("out.shape : ", out.shape)
        for detection in out:
            # if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            # if scores[classId]>confThreshold:
            confidence = scores[classId]
            # if detection[4] > tconfThreshold:
            #     print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
            if confidence > tconfThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)

                if not first_pass:
                    center_x = center_x + rectangle_of_tomatoes[l][0]
                    center_y = center_y + rectangle_of_tomatoes[l][1]

                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    k = 0
    for i in indices:
        i = i
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        #objVector.append(Objection(Rcet(left, left + width, top, top + height), classIds[i]))
        # print("objVector: =======",objVector)
        drawPred(result_image, classIds[i], confidences[i], left, top, left + width, top + height)
        if not first_pass:
            rectangle_of_defects.append([int(left), int(top), int(width), int(height)])
        if first_pass:
            crop_imgs.append(color_image[top:top + height, left:left + width])
            rectangle_of_tomatoes.append([int(left), int(top), int(width), int(height)])
            #center_bounding_box.append([int(left + width / 2), int(top + height / 2)])
            glob_center_bbox.append([int(left + width / 2), int(top + height / 2)])
            k += 1
    print("Number_Tomato= ", k)
    print("center_bounding_box= ", glob_center_bbox)
    #print("rectangle_of_tomatoes= ", rectangle_of_tomatoes)
    return glob_center_bbox

print(sys.version) # get python version
cnt_im=0

# start pipeline
# get intrinsics

# store coordinates for sorting and write in file
tomato_coords_list = []

#clean files
cleanFile('tomato_coords_list.txt')
cleanFile('tomato_coords_list_sorted.txt')

#while True:
# get path to rgb image from zivid
with zivid.Application():
    app = zivid.Application()
    camera = app.connect_camera()
    # settings for capture assistant
    suggest_settings_parameters = SuggestSettingsParameters(max_capture_time=datetime.timedelta(milliseconds=1200),ambient_light_frequency=SuggestSettingsParameters.AmbientLightFrequency.none,)
    settings = zivid.capture_assistant.suggest_settings(camera, suggest_settings_parameters)
    # save image
    with camera.capture(settings) as frame:
        frame.save("result.zdf")
    # get path to zivid image
    filename_zdf = "/home/robolab/Zivid/tompabotti_zivid/TompabottiWithZivid/Tompabotti/Tomato_project/Program/result.zdf"
    #filename_zdf = Path() / get_sample_data_path() / "result.zdf"

    print(f"Reading {filename_zdf} point cloud")
    frame = zivid.Frame(filename_zdf)
    # get point cloud from frame
    point_cloud = frame.point_cloud()
    xyz = point_cloud.copy_data("xyz")
    rgba = point_cloud.copy_data("rgba")
    snr = frame.point_cloud().copy_data("snr")
    # discomment for image display
    #display_rgb(rgba[:, :, 0:3], block=False)

    #display_depthmap(xyz, block=True)

    #display_pointcloud(xyz, rgba[:, :, 0:3])

    # save 2d rgb image
    file_path = "/home/robotics/rs2_to_zivid/Tompabotti/Tomato_project/Program/result.zdf"
    #file_path = Path(user_options.result)
    _convert_2_2d(point_cloud, file_path + ".png")
    # get np array
    flat_pc = _flatten_point_cloud(point_cloud)
    print("Converting to BGR image in OpenCV format")
    bgr = _point_cloud_to_cv_bgr(point_cloud)
    print("Converting to Depth map in OpenCV format")
    z_color_map = _point_cloud_to_cv_z(point_cloud)

filename_zdf_png = "/home/robotics/rs2_to_zivid/Tompabotti/Tomato_project/Program/result.zdf.png"

# take color image from zdf
color_image = bgr
cnt_im+=1

# take depth image from zdf
depth_image = z_color_map

if show:
    plt.imshow(color_image)#, alpha=0.6)
    #plt.imshow(depth_image, alpha=0.8)
    # plt.imshow(colorized_depth, alpha=0.6)
    plt.show()
    #key = cv.waitKey(1)

qualify_image = color_image
depth_picture = depth_image
#pipeline.stop()

qualify_depth = depth_picture.copy()
qualify_result = qualify_image.copy()


first_pass = True
# Create necessary lists
crop_imgs = [] #crop images for each single tomato
rectangle_of_tomatoes = [] #coordinate of each tomato
rectangle_of_defects = [] #coordinate of each defect
nondefect_tomatoes = [] #list of tomatoes dont have defects on it
bad_tomatoes = [] #list of bad tomatoes

# Neural Net > Deep Neural Network
#############################################################################################
net = cv.dnn.readNetFromDarknet(tmodelConfiguration, tmodelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# timg = cv.imread(timgPath)
timg = qualify_image

# Create a 4D blob from a frame.
blob = cv.dnn.blobFromImage(timg, 1 / 255, (tinpWidth, tinpHeight), [0, 0, 0], 1, crop=False)

# Sets the input to the network
net.setInput(blob)

# Runs the forward pass to get output of the output layers
outs = net.forward(getOutputsNames(net))
#objVector=[]
# Remove the bounding boxes with low confidence
p = postprocess(timg, qualify_result, outs, tconfThreshold, tnmsThreshold)
#print("postprocess center of bbox = ", p)
#print("obj vector: ", objVector)

print("#########################################################################################")


tomato_coords_tuple = coords(rectangle_of_tomatoes, qualify_result, qualify_image)
#print("tomato_coords_tuple=",tomato_coords_tuple)

# getting tomato coordinates as x, y, z from zivid
# tomato coords tuple is used to access pixel values for zivid
print("ææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææÆÆÆÆÆ")
zivid_tomatoes_list = []
if len(tomato_coords_tuple) != 0:
    zivid_tomatoes = get_zivid_coords( tomato_coords_tuple[1],  tomato_coords_tuple[0], 1)
    print("tomatoes from zivid: ", zivid_tomatoes)

    zivid_tomatoes_list.append(zivid_tomatoes)
    print("zivid_tomatoes_list:--------------------------------- ", zivid_tomatoes_list)
        # move to the next element of the center of bounding box list (p)
    print("000000000000000000000000000000000000000000000000000000000000000000000000Ä")
    #for i in range(len(zivid_tomatoes)):
    #    zivid_tomatoes_list.append(list(zivid_tomatoes))
    #    print("zivid_tomatoes_list: ", zivid_tomatoes_list)

    #height= tomato_coords_tuple[1]   # y pixel
    #width = tomato_coords_tuple[0]   # x pixel
    # save if not empty and with detected Z > 0
    if len(zivid_tomatoes) > 0:              # and zivid_tomatoes[2]>0 and zivid_tomatoes[2]<750:

        #prePoint = np.array([zivid_tomatoes[0], zivid_tomatoes[1]])
        prePoint = np.array(zivid_tomatoes)
        new_PT = Position_tansform(prePoint)
        #print("New_PT", new_PT)
        print("zivid appended: ", zivid_tomatoes)
        # write to file
        # save coordinates to list
        #for i in p:
        saveCoordsFile((zivid_tomatoes), 'tomato_coords_list.txt', 'a')

# Show result with the cutting points
if show:
    show_process_image('Qualify result', qualify_result)

result_image = qualify_image #getframe()
# weight_image = cv.imread(full_path)
result = result_image.copy()
if show:
    show_process_image('Start weight', result)

key = cv.waitKeyEx(5000)
#if key == ord('q'):
 #   saveCoordsFile(tomato_coords_list,)
    #break
#print(readCoordsFile())
#cleanFile('tomato_coords_list_sorted.txt')
# sort tomato_coords_list.txt and get tuples of coordinates
listOfSorted = readCoordsFile('tomato_coords_list.txt')   # return listTuplesofCoords

#if len(listOfSorted) > 0 and Obj.PCL[2] / 1000 < 0.750:

saveCoordsFileFromList(listOfSorted, 'tomato_coords_list_sorted.txt', 'w')
#length = len(listOfSorted)
#print("length=",length)
#if len(listOfSorted) > 0:
#    saveCoordsFileFromList([listOfSorted, 'tomato_coords_list_sorted.txt', 'w')
#arr = readForArray('tomato_coords_list_sorted.txt')
#print("arr=", arr)

# if cv.waitKeyEx(1) & 0xFF == ord('q'):
#     break
