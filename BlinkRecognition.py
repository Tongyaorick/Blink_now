# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:26:29 2019

@author: TOA2SZH
"""

import dlib         # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 Numpy
import cv2          # 图像处理的库 OpenCv
import os
import shutil
from skimage import io
import csv
import pandas  as pd
import time
from scipy.spatial import distance
from imutils import face_utils


class BlinkRecognition:
    """In this class,you are able to save figures and detect blinks """
    def __init__(self):
        # Dlib 预测器
        #Dlib 正向人脸检测器 / frontal face detector
        self.detector = dlib.get_frontal_face_detector()
        #Dlib 68 点特征预测器 / 68 points features predictor
        self.predictor = dlib.shape_predictor('data/models/shape_predictor_68_face_landmarks.dat')
        #128维人脸特征向量检测器
        self.faceres = dlib.face_recognition_model_v1('data/models/dlib_face_recognition_resnet_model_v1.dat')
        #该路径存放拍摄下来的图片
        self.path_img_dir = "data/faces_from_camera/"
        #该路径存放保存的csv文件
        self.path_csv_dir = "data/csv_from_camera/"
        #存放所有特征均值的csv路径
        self.path_csv_features_all = "data/features_all.csv"
        #计算人脸个数
        self.person_count = 0
        #匿名人个数
        self.annoymous_count = 0
        # 将人脸计数器初始化
        self.count_single_person = 0
        #字体
        self.font = cv2.FONT_HERSHEY_COMPLEX
        #初始化Person的图片文件路径
        self.current_PERSON_face_path = 0
        #已经存储的特征数列
        self.features_known_arr    = []
        #捕获到的人脸特征数列
        self.features_captured_arr = []
        # 提示输入名字信号
        self.input_name_signal = 0
        # EAR参数阈值
        self.EYE_AR_THRESH = 0.3
        # 当EAR小于阈值时，接连多少帧一定发生眨眼动作
        self.EYE_AR_CONSEC_FRAMES = 3
        # 对应眼部特征点的序号
        self.RIGHT_EYE_START = 37 - 1   #36
        self.RIGHT_EYE_END   = 42 - 1     #41
        self.LEFT_EYE_START  = 43 - 1    #42
        self.LEFT_EYE_END    = 48 - 1      #47
        #定义眨眼计数器
        self.blink_counter = {}
        self.frame_counter = {}
        
        
        
        
        
    
    
    def _create_camera(self):
        """
        In this method,you create a camera instance and set its parameters
        """
        # 创建 cv2 摄像头对象，0表示使用摄像头，否则输入视频路径
        self.cap = cv2.VideoCapture(0)
        # cap.set(propId, value)
        # 设置视频参数，propId 设置的视频参数，value 设置的参数值
        #CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream 视频流中帧的宽度
        self.cap.set(3,480)
        
    #清理图片文件夹和csv文件夹    
    def _clean_dirs(self):
        ## 清空对应的文件夹
        # 清空拍摄图片的文件夹
        folder_reader = os.listdir(self.path_img_dir)
        for file in folder_reader:
            shutil.rmtree(self.path_img_dir+file)
        # 清空csv文件的文件夹
        csv_reader = os.listdir(self.path_csv_dir)
        for file in csv_reader:
            os.remove(self.path_csv_dir+file)
    
    
    def _CREATE_NEW_IMG_PATH_TEXT(self):
        print("Creating new image path for you")
        print("Please wait",end='')
        for i in range(4):
            print(">",end='')
            time.sleep(0.2)
            print(">")
       

    def _PUT_TEXT(self,img_read,rectangles):
        #页面显示
        #cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
        #@param org Bottom-left corner of the text string in the image. . 
        #@param fontFace Font type, see #HersheyFonts. . 
        #@param fontScale Font scale factor that is multiplied by the font-specific base size. . 
        #@param color Text color. . @param thickness Thickness of the lines used to draw a text. . 
        #@param lineType Line type. See #LineTypes . @param bottomLeftOrigin When true, the image data origin is at the bottom-left corner. Otherwise, . it is at the top-left corner.
        #人脸数说明
        cv2.putText(img_read,"Faces: "+str(len(rectangles)),(20,100),self.font,0.8,(0,98,73),1,cv2.LINE_AA)
        # 添加说明
        cv2.putText(img_read,"Face Register",     (20,40 ),self.font,1  ,(0,168,176),1,cv2.LINE_AA)
        cv2.putText(img_read,"N: New face folder",(20,350),self.font,0.8,(0,168,176),1,cv2.LINE_AA)
        cv2.putText(img_read,"S: Save face",      (20,400),self.font,0.8,(0,168,176),1,cv2.LINE_AA)
        cv2.putText(img_read,"Q: Quit",           (20,450),self.font,0.8,(0,168,176),1,cv2.LINE_AA)
                    
            

    def _PRESS_N(self,u_i):
        if (u_i & 0xFF) == ord('n') or (u_i & 0xFF) == ord('N'):
            recreate_signal = 1
            self.person_count += 1
            print(">>>>>>>>>>>>> <<<<<<<<<<<<<")
            person_name = input("Please input you NAME below \n")
            self._CREATE_NEW_IMG_PATH_TEXT()
            #为用户输入的新名字创建一个新路径
            self.current_PERSON_face_path = self.path_img_dir+"person_"+str(self.person_count)+"_"+person_name
            #遍历当前存放图片路径下的每一个文件夹
            for img_sub_dir in os.listdir(self.path_img_dir):
                #如果存在重名的，抛出信息并删除原有的所有重名文件夹
                if person_name == img_sub_dir[9:]:
                    self.person_count -= 1
                    shutil.rmtree(self.path_img_dir + img_sub_dir)
                    print("Found double name directory:",img_sub_dir)
                    time.sleep(1)
                    print("DELETING double name directory!!!")
                    #新建图片文件夹
                    os.makedirs(self.path_img_dir + img_sub_dir)
                    recreate_signal = 0
                    self.current_PERSON_face_path = self.path_img_dir + img_sub_dir
                    break
            if recreate_signal != 0:
                #新建图片文件夹
                os.makedirs(self.current_PERSON_face_path)
                print("Creating new image path @ ",self.current_PERSON_face_path)
            # 将人脸计数器清零
            self.count_single_person = 0
        else:
            pass
        
    ##########################################
    ########                          ########
    ########  PART1 face_recognition  ########
    ########                          ########
    ##########################################                
    def face_recognition_main(self):
        #运行脸部特征识别并保存时，先检查img文件夹和csv文件夹是否匹配
        img_dir_users = len(os.listdir(self.path_img_dir))
        csv_dir_users = len(os.listdir(self.path_csv_dir))
        #若img文件夹和csv文件夹不匹配，则直接报错
        if img_dir_users != csv_dir_users:
            raise ImportError("Image directory and csv directory do not match!!!")
        #否则，文件夹里已有人数即为当前的person_count
        else:
            self.person_count = img_dir_users
            
        #创建摄像头对象    
        self._create_camera()
        #当摄像头处于开启状态时
        while self.cap.isOpened():
            self.input_name_signal -= 1
            #flag 是 True/False img_read是RGB图像
            flag,img_read = self.cap.read()
            #设置图像等待用户输入的延迟,kk为用户输入(单位是毫秒)
            user_input = cv2.waitKey(33)
            #将RGB图像转化为灰度图像
            img_gray = cv2.cvtColor(img_read,cv2.COLOR_RGB2GRAY)
            #该矩形存放人脸位置
            rectangles = self.detector(img_gray,1)
            
            
                
            ############################################################################################    
            #如果监测到了多于一张人脸，则抛出异常提示
            if len(rectangles) > 1:
                cv2.putText(img_read,"MORE THAN 1 FACE",(200,300),self.font,1,(226,0,21),3,cv2.LINE_AA)
            #当只检测到一张脸时
            elif len(rectangles) != 0:
                #判定用户输入是否为 n 或者 N
                ################
                ##  #   #     ##
                ##  ##  #     ##
                ##  # # #     ## 
                ##  #  ##     ##
                ##  #   #     ##
                ##            ##
                ################
                ################
                self._PRESS_N(user_input)
                #检测并计算矩形框
                rec = rectangles[0]
                #得出矩形框的左上角位置和右下角位置
                position_left_top     = tuple([rec.left() , rec.top()   ])
                position_right_bottom = tuple([rec.right(), rec.bottom()])
                #计算矩形框的高度和宽度
                height = rec.bottom()- rec.top()
                width  = rec.right() - rec.left()
                #cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
                cv2.rectangle(img_read,position_left_top,position_right_bottom,(0,142,207),3)
                #当按下S时
                if (user_input & 0xFF) == ord('s') or (user_input & 0xFF) == ord('S'):
                    #根据人脸大小初始化空的图像
                    img_blank = np.zeros((height, width, 3), np.uint8)
                    #若果之前有保存过名字
                    if self.current_PERSON_face_path != 0:
                        self.count_single_person += 1
                        #提取原有图像中的人脸部分保存入img_blank部分
                        for hi in range(height):
                            for hj in range(width):
                                img_blank[hi][hj] = img_read[rec.top()+hi][rec.left()+hj]
                        #将脸部图片存入本地
                        single_img_path = self.current_PERSON_face_path+"/image_face_"+str(self.count_single_person)+".jpg"
                        cv2.imwrite(single_img_path,img_blank)
                        print("Found directory ",self.current_PERSON_face_path)
                        print("Saving images...","/image_face_"+str(self.count_single_person)+".jpg")
                    #如果用户没有在保存前输入新名字,则要求用户输入名字
                    else:
                        self.input_name_signal = 5
                        print("Press N to input Name first!")
                        continue
                        #新建图片文件夹
#                        try:
#                            os.makedirs(self.current_PERSON_face_path)
#                        except:
#                            current_DIRECTORY = list(os.listdir(self.path_img_dir))
#                            existing_count = 0
#                            for folder_name in current_DIRECTORY:
#                                if self.current_PERSON_face_path in self.path_img_dir+folder_name:
#                                    existing_count += 1
#                            existing_count += 1
#                            self.current_PERSON_face_path = self.current_PERSON_face_path+'_doublename_'+str(existing_count)
#                            os.makedirs(self.current_PERSON_face_path)
                    
                    
                    
            
            if self.input_name_signal > 0:
                cv2.putText(img_read,"Name First",(150,200),self.font,2,(226,0,21),3,cv2.LINE_AA)
            ##############################################################################################
            #页面显示
            self._PUT_TEXT(img_read,rectangles)
            #若是用户输入q/Q,则退出程序
            if (user_input & 0xFF) == ord('q') or (user_input & 0xFF) == ord('Q'):
                break
            #展示此刻的图片
            cv2.imshow("FACE_RECOGNITION",img_read)
        #退出上述循环后释放摄像头
        self.cap.release()
        #删除建立的窗口
        cv2.destroyAllWindows()
        
        
    ##########################################
    ########                          ########
    ########  PART2 save_to_csv_file  ########
    ########                          ########
    ##########################################
    #读取单张照片，返回一个128维的特征向量    
    def return_128d_features(self,path_img):
        img = io.imread(path_img)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        dets = self.detector(img_gray,1)
        print("Detecting face images:",path_img)
        if len(dets) != 0:
            shape = self.predictor(img_gray,dets[0])
            #使用resNet获取128维的人脸特征向量a
            face_description = self.faceres.compute_face_descriptor(img_gray,shape)
        else:
            face_description = 0
            print("No face images detected ")
        return face_description
    
    
    #将文件夹中的照片提取出来， 写入csv文件
    #读取单人文件夹，写入单个csv文件夹
    #每张照片写入一行
    def write_into_csv(self,path_faces_person_X,path_csv_file):
        dir_pics = os.listdir(path_faces_person_X)
        with open(path_csv_file,'w',newline='') as csvfile:
            writer = csv.writer(csvfile)
            for pic in dir_pics:
                print("Loading face images: "+path_faces_person_X+'/'+pic)
                features_128d = self.return_128d_features(path_faces_person_X+'/'+pic)
                if features_128d == 0:
                    pass
                else:
                    writer.writerow(features_128d)

    #读取整个image文件夹，把每个文件夹写成一个csv        
    def read_and_write_person_all_pics(self):
        #读取某人的所有脸的数据,写入对应的csv文件
        faces = os.listdir(self.path_img_dir)
        for person in faces:
            print("Handling ",person)
            person_img_dir = self.path_img_dir+person
            person_csv_file = self.path_csv_dir+person+'.csv'
            self.write_into_csv(person_img_dir,person_csv_file)            
      
        
        
    #从csv中读取数据， 计算128d特征的均值
    #读取单个csv文件,返回一个各维度数值均值组成的list
    def comput_the_mean(self,path_csv_file):
        #128列的特征
        column_names = ["feature_"+str(i+1) for i in range(128)]
        #使用pandas读取csv
        tmp_data = pd.read_csv(path_csv_file,header=None,names=column_names)
        feature_mean = [np.mean(np.array(tmp_data["feature_"+str(i+1)])) for i in range(128)]
        return feature_mean
    
    
    #将存储csv文件的csv文件夹下的所有csv文件，分别计算特征均值并放入一个features_all.csv当中
    def run_comput(self):
        with open(self.path_csv_features_all,'w+',newline = '') as csvfile:
            writer = csv.writer(csvfile)
            csv_reader = os.listdir(self.path_csv_dir)
            for csv_file in csv_reader:
                current_csv_file = self.path_csv_dir+csv_file
                feature_mean = self.comput_the_mean(current_csv_file)
                print("Computing the feature mean of ",csv_file)
                writer.writerow(feature_mean)
    

    #将文件保存为csv格式
    def save_2_csv_main(self):
        self.read_and_write_person_all_pics()
        self.run_comput()
    
    
    
        
    ##########################################
    ########                          ########
    ########  PART3 face_recognition  ########
    ########                          ########
    ##########################################
    # 返回一张图像多张人脸的 128D 特征
    # 存入face_des的list中，每个元素代表一张人脸
    def get_128d_features(self,img_gray):
        dets = self.detector(img_gray, 1)
        if len(dets) != 0:
            face_des = []
            for i in range(len(dets)):
                shape = self.predictor(img_gray, dets[i])
                #facerec.compute_face_descriptor(img_gray,shape)  会返回一个128维的特征向量
                face_des.append(self.facerec.compute_face_descriptor(img_gray,shape))
        else:
            face_des = []
        return face_des
    
    
    
    
    
    
    ########################################################################################################
    #计算两个向量间的欧式距离并返回diff或者same
    def return_euclidean_distance(self,feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        #print("E_distance: ",dist)
        if dist > 0.4:
            return "diff"
        else:
            return "same"
    
    
    # 用来存放所有录入人脸特征的数组
    #features_all的每一行变为features_know_arr的每一个元素
    def cal_known_faces(self):
        try:
            #features_all的dataframe格式
            self.features_all = pd.read_csv(self.path_csv_features_all,header=None)
        except:
            raise ImportError("Please run the method save_2_csv first!!!")
        #初始化
        self.features_known_arr = []
        # known faces
        for i in range(self.features_all.shape[0]):
            features_someone_arr = []
            for j in range(len(self.features_all.iloc[i,:])):
                features_someone_arr.append(self.features_all.iloc[i,j])
            #    print(features_someone_arr)
            self.features_known_arr.append(features_someone_arr)
        print("Faces in Database：", len(self.features_known_arr))
        
        
        
    # 计算EAR --- Eye_aspect_ratio
    # 输入眼部特征，输出 eye_aspect_ratio特征
    def eye_aspect_ratio(self,eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A+B)/(2.0*C)
        return ear
    
    
    
    def face_recognition_and_blink_recognition_main(self):
        #先获取所有的已经录入的人脸特征数组
        self.cal_known_faces()
        #创建摄像头实例对象
        self._create_camera()
        current_namelist = []
        #初始化眨眼计数器
        self.blink_counter = {}
        self.frame_counter = {}
        while self.cap.isOpened():
            #flag是True/False ,img_read是当前帧的RGB图像
            flag,img_read = self.cap.read()
            user_input = cv2.waitKey(33)
            # 把RGB图像转换为灰度图像
            img_gray = cv2.cvtColor(img_read,cv2.COLOR_RGB2GRAY)
            faces = self.detector(img_gray,1)
            #写上提示字体
            cv2.putText(img_read,"Q: Quit",(20,450),self.font,0.8,(0,98,73),1,cv2.LINE_AA)
            #存储所有的人脸名字和对应的位置
            position_namelist = []
            name_namelist     = []
            if len(faces) != 0:
                self.features_captured_arr = []
                #把当前捕获到的人脸特征,以及名字和坐标特征存起来
                for i in range(len(faces)):
                    shape = self.predictor(img_read,faces[i])
                    self.features_captured_arr.append(self.faceres.compute_face_descriptor(img_read, shape))
                    #把特征转换为numpy格式的点
                    points = face_utils.shape_to_np(shape)
                    # 让人名跟随在矩形框的下方
                    # 确定人名的位置坐标
                    # 先默认所有人不认识，是 unknown
                    name_namelist.append("Unknown_"+str(i+1))
                    self.blink_counter["Unknown_"+str(i+1)] = 0
                    self.frame_counter["Unknown_"+str(i+1)] = 0
                    # 每个捕获人脸的名字坐标
                    position_namelist.append(tuple([faces[i].left(),int(faces[i].bottom()+(faces[i].bottom()-faces[i].top())/4)]))
                
                    for k in range(len(self.features_known_arr)):
                        # 将某张人脸与存储的所有人脸数据进行比对
                        #print("Comparing with person_",str(k+1))
                        compare_result = self.return_euclidean_distance(self.features_captured_arr[i], self.features_known_arr[k])
                        if compare_result == 'same':
                            print("Found person_%d similar with you!"%(k+1))
                            name_namelist[i] = 'person_'+str(k+1)
                            dir_csvs = os.listdir(self.path_csv_dir)
                            if str(k+1) in dir_csvs[k]:
                                name_namelist[i] = dir_csvs[k][9:-4]
                                self.blink_counter.pop("Unknown_"+str(i+1))
                                self.frame_counter.pop("Unknown_"+str(i+1))
                                if name_namelist[i] not in self.blink_counter.keys():
                                    self.blink_counter[name_namelist[i]] = 0
                                    self.frame_counter[name_namelist[i]] = 0
                            break
                                
                    #提取左眼、右眼部分
                    leftEye  = points[self.LEFT_EYE_START:self.LEFT_EYE_END+1]
                    rightEye = points[self.RIGHT_EYE_START:self.RIGHT_EYE_END + 1]
                    #计算左右眼的EAR
                    leftEAR  = self.eye_aspect_ratio(leftEye)
                    rightEAR = self.eye_aspect_ratio(rightEye)
                    print("Left EAR  :{}".format(leftEAR))
                    print("Right EAR :{}".format(rightEAR))
                    #平均EAR
                    ear = (leftEAR + rightEAR) / 2.0
                    #得到眼部跟踪框
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    # 绘制眼部跟踪框
                    # drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]])
                    cv2.drawContours(img_read, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(img_read, [rightEyeHull], -1, (0, 255, 0), 1)
                    #计算眨眼的逻辑
                    if ear<self.EYE_AR_THRESH:
                        self.frame_counter[name_namelist[i]] = self.frame_counter[name_namelist[i]]+1
                    else:
                        if self.frame_counter[name_namelist[i]] >= self.EYE_AR_CONSEC_FRAMES:
                            self.blink_counter[name_namelist[i]] = self.blink_counter[name_namelist[i]]+1
                        self.frame_counter[name_namelist[i]] = 0
                    ###绘制框
                    cv2.putText(img_read,"Blinks:{}".format(self.blink_counter[name_namelist[i]]),(10+100*i,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,86,145),2)
                    cv2.putText(img_read,"EAR:{:.2f}".format(ear),(300+10*i,30+20*i),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,86,145),2)

                    if 'Unknown' not in name_namelist[i]:
                        print("Found name:",name_namelist[i])
#                    dir_csvs = os.listdir(self.path_csv_dir)
#                    result = "big" in dir_csvs[i]
#                    result1 = "small" in dir_csvs[i]
#                    if result == True:
#                        cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 1.0, (0, 255, 255), 1, cv2.LINE_AA)
#                    elif result1 == True:
#                        cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
#                    else:
#                        cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

                    if self.blink_counter[name_namelist[i]] > 0:
                        cv2.rectangle(img_read,tuple([faces[i].left(),faces[i].top()]),tuple([faces[i].right(),faces[i].bottom()]),(120,190,32),5)
                        cv2.putText(img_read,name_namelist[i],position_namelist[i],self.font,1,(120,190,32),5,cv2.LINE_AA)
                    else:
                        cv2.rectangle(img_read,tuple([faces[i].left(),faces[i].top()]),tuple([faces[i].right(),faces[i].bottom()]),(0,255,255),2)
                        cv2.putText(img_read,name_namelist[i],position_namelist[i],self.font,1,(0,255,255),2,cv2.LINE_AA)
            if name_namelist != current_namelist:
                print("Name list now:", name_namelist, "\n")
                current_namelist = name_namelist.copy()
            else:
                pass 
            
        
            cv2.putText(img_read,"Face Register",          (20,40 ),self.font,1,(0,168,176),1,cv2.LINE_AA)
            cv2.putText(img_read,"Faces: "+str(len(faces)),(20,100),self.font,1,(0,168,176),1,cv2.LINE_AA)
            
            # 按下 q 键退出1
            if  (user_input & 0xFF) == ord('q') or (user_input & 0xFF) == ord('Q'):
                break
            # 窗口显示
            cv2.imshow("camera", img_read)
        # 释放摄像头
        self.cap.release()
        # 删除建立的窗口
        cv2.destroyAllWindows()
        
    def full_run(self):
        self.face_recognition_main()
        self.save_2_csv_main()
        self.face_recognition_and_blink_recognition_main()
        
    def blink_recognition_run(self):
        self.face_recognition_and_blink_recognition_main()

        
        
