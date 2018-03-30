import os
import sys
import cv2
import numpy as np
from PyQt4 import QtGui
from PyQt4 import QtCore
from libs.ustr import ustr
import libs.annoCvt as annoCvt
import libs.bgGen as bgGen
import libs.dataMerge as dataMerge
import libs.CascadeDetector as CD

__appname__ = 'trainDetector'

class Ui_Dialog(QtGui.QDialog):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        #global c
        #c = dbConnection

class Example(QtGui.QMainWindow):
    
    def __init__(self):
        super(Example, self).__init__()

        self.defaultSourceDir = None
        self.defaultImageDir = '.'
        self.defaultTestDir = None
        self.filePath = '.'
        self.initUI()
        self.AnnotationFile = None
        self.BackgroundFile = None
        
    def initUI(self):             
        # Image widget 
        hbox = QtGui.QHBoxLayout(self)
        image_size = (360, 270)

        init_image = cv2.imread('./candy.jpg')
        init_image = cv2.cvtColor(init_image, cv2.COLOR_BGR2RGB)
        init_image = cv2.resize(init_image, image_size)
        init_q_image = QtGui.QImage(init_image.data,init_image.shape[1], init_image.shape[0], QtGui.QImage.Format_RGB888)
        init_pixmap = QtGui.QPixmap()
        init_pixmap.convertFromImage(init_q_image)

        self.imagelabel = QtGui.QLabel(self)
        self.imagelabel.setPixmap(init_pixmap)

        hbox.addWidget(self.imagelabel)
        self.setLayout(hbox)
        self.imagelabel.setGeometry(200, 50, image_size[0], image_size[1])
        #self.imagelabel.move(300, 50)
        
        # Actions
        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)        
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.qApp.quit)

        setsourcedirAction = QtGui.QAction('&Change source dir', self)
        setsourcedirAction.triggered.connect(self.changeSourcedirDialog)

        setimagedirAction = QtGui.QAction('&Change image dir', self)
        setimagedirAction.triggered.connect(self.changeImagedirDialog)

        readannoAction = QtGui.QAction('&Read Annotation', self)
        readannoAction.triggered.connect(self.openAnnotationDialog)

        readbgAction = QtGui.QAction('&Read Background', self)
        readbgAction.triggered.connect(self.openBackgroundDialog)

        # Buttons
        button_x_offset = 20
        button_y_list = [20, 70 , 120, 170, 220, 270, 320]
        setimagedirButton = QtGui.QPushButton('image dir', self)
        setimagedirButton.clicked.connect(self.changeImagedirDialog)
        setimagedirButton.resize(setimagedirButton.sizeHint())
        setimagedirButton.move(button_x_offset, button_y_list[0])

        setsourcedirButton = QtGui.QPushButton('source dir', self)
        setsourcedirButton.clicked.connect(self.changeSourcedirDialog)
        setsourcedirButton.resize(setsourcedirButton.sizeHint())
        setsourcedirButton.move(button_x_offset, button_y_list[1])

        settestdirButton = QtGui.QPushButton('test dir', self)
        settestdirButton.clicked.connect(self.changeTestdirDialog)
        settestdirButton.resize(settestdirButton.sizeHint())
        settestdirButton.move(button_x_offset, button_y_list[2])

        mergedataButton = QtGui.QPushButton('merge datasets', self)
        mergedataButton.clicked.connect(self.Mergedata)
        mergedataButton.resize(mergedataButton.sizeHint())
        mergedataButton.move(button_x_offset, button_y_list[3])

        genfilesButton = QtGui.QPushButton('generate files', self)
        genfilesButton.clicked.connect(self.genFiles)
        genfilesButton.resize(genfilesButton.sizeHint())
        genfilesButton.move(button_x_offset, button_y_list[4])

        trainButton = QtGui.QPushButton('train detector', self)
        trainButton.clicked.connect(self.trainDetector)
        trainButton.resize(trainButton.sizeHint())
        trainButton.move(button_x_offset, button_y_list[5])

        testButton = QtGui.QPushButton('test detector', self)
        testButton.clicked.connect(self.testDetector)
        testButton.resize(testButton.sizeHint())
        testButton.move(button_x_offset, button_y_list[6])

        # menuBar
        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)
        fileMenu.addAction(setsourcedirAction)
        fileMenu.addAction(readannoAction)
        fileMenu.addAction(readbgAction)
        fileMenu.addAction(setimagedirAction)
        
        self.setGeometry(600, 600, 600, 400)
        self.setWindowTitle(__appname__)    
        self.show()
    
    def changeImagedirDialog(self, _value=False):
        if self.defaultImageDir is not None:
            path = ustr(self.defaultImageDir)
        else:
            path = '.'
        
        dirpath = ustr(QtGui.QFileDialog.getExistingDirectory(self,
                                                       '%s - Use image in the directory' % __appname__, path,  QtGui.QFileDialog.ShowDirsOnly
                                                       | QtGui.QFileDialog.DontResolveSymlinks))
        if dirpath is not None and len(dirpath) > 1:
            self.defaultImageDir = dirpath

        self.statusBar().showMessage('%s . Use the images in  %s' %
                                     ('Change image folder', self.defaultImageDir))
        self.statusBar().show()

        self.AnnotationFile = str(self.defaultImageDir+'/info.dat')
        self.BackgroundFile = str(self.defaultImageDir+'/bg.txt')

    def changeSourcedirDialog(self, _value=False):
        if self.defaultSourceDir is not None:
            path = ustr(self.defaultSourceDir)
        else:
            path = '.'

        dirpath = ustr(QtGui.QFileDialog.getExistingDirectory(self,
                                                       '%s - Set source dataset to the directory' % __appname__, path,  QtGui.QFileDialog.ShowDirsOnly
                                                       | QtGui.QFileDialog.DontResolveSymlinks))

        if dirpath is not None and len(dirpath) > 1:
            self.defaultSourceDir = dirpath

        self.statusBar().showMessage('%s . Data will be read from %s' %
                                     ('Change source folder', self.defaultSourceDir))
        self.statusBar().show()

    def changeTestdirDialog(self, _value=False):
        if self.defaultTestDir is not None:
            path = ustr(self.defaultTestDir)
        else:
            path = '.'

        dirpath = ustr(QtGui.QFileDialog.getExistingDirectory(self,
                                                       '%s - Set test dataset to the directory' % __appname__, path,  QtGui.QFileDialog.ShowDirsOnly
                                                       | QtGui.QFileDialog.DontResolveSymlinks))

        if dirpath is not None and len(dirpath) > 1:
            self.defaultTestDir = dirpath

        self.statusBar().showMessage('%s . Test Data will be read from %s' %
                                     ('Change test folder', self.defaultTestDir))
        self.statusBar().show()

    def Mergedata(self, _value=False):
        # Merge source dataset into image directory
        dataset_dir = self.defaultSourceDir+'/' #"/home/yxiao1996/data/balls/1-25/"
        target_dir = self.defaultImageDir+'/' # "/home/yxiao1996/data/balls/"

        self.statusBar().showMessage('%s . Merging dataset from %s to %s' %
                                     ('Merge datasets', self.defaultSourceDir, self.defaultImageDir))
        self.statusBar().show()

        offset = dataMerge.getOffset(target_dir)
        print ("pos offset", offset)
        xml_itor = dataMerge.getItor(dataset_dir+"Anno/")
        for xml_fn in xml_itor:
            root = dataMerge.getRoot(xml_fn)
            filename = root.find("filename").text
            dataMerge.moveImg(filename, offset, target_dir, dataset_dir)
            dataMerge.moveAnno(root, offset, target_dir)

        neg_offset = dataMerge.getOffset(target_dir, pos=False)
        print ("neg offset", neg_offset)
        neg_itor = dataMerge.getNegItor(dataset_dir)
        for path in neg_itor:
            filename = path.split('/')[-1]
            dataMerge.moveImg(filename, neg_offset, target_dir, dataset_dir, pos=False)

    def openAnnotationDialog(self, _value=False):
        path = os.path.dirname(ustr(self.filePath))\
            if self.filePath else '.'
        
        filters = "Open Annotation .dat file (%s)" % ' '.join(['*.dat'])
        filename = ustr(QtGui.QFileDialog.getOpenFileName(self,'%s - Choose a .dat file' % __appname__, path, filters))
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
        self.AnnotationFile = filename

    def openBackgroundDialog(self, _value=False):
        path = os.path.dirname(ustr(self.filePath))\
            if self.filePath else '.'
        
        filters = "Open Background .txt file (%s)" % ' '.join(['*.txt'])
        filename = ustr(QtGui.QFileDialog.getOpenFileName(self,'%s - Choose a .txt file' % __appname__, path, filters))
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
        self.BackgroundFile = filename
        
    def genFiles(self, _value=False):
        # 1: Convert annotations
        # Set annotation directory and output file
        anno_dir = str(self.defaultImageDir+'/Anno/')
        out_filename = str(self.defaultImageDir+'/info.dat')
        out_file = annoCvt.openOut(out_filename)
        filename_itor = annoCvt.getItor(anno_dir)

        # Convert annotation from xml to info.dat
        for filename in filename_itor:
            root = annoCvt.getRoot(filename)
            annoCvt.convert(root, out_file)
        out_file.close()

        self.statusBar().showMessage('%s . Reading annotations from %s/Anno/' %
                                     ('Generate training files', self.defaultSourceDir))
        self.statusBar().show()

        # 2: generate background file
        neg_dir = str(self.defaultImageDir+'/neg/')
        out_filename = str(self.defaultImageDir+'/bg.txt')

        # Generate bg.txt
        itor = bgGen.getItor(neg_dir)
        out_file = bgGen.openOut(out_filename)

        for filename in itor:
            bgGen.convert(filename, out_file)

        out_file.close()

        self.statusBar().showMessage('%s . Reading negative images from %s/neg/' %
                                     ('Generate training files', self.defaultSourceDir))
        self.statusBar().show()

    def trainDetector(self, _value=False):
        """using opencv_createsamples to generate positive samples"""
        print "generating positive data..."
        os.chdir(self.defaultImageDir)

        cmmd = str("opencv_createsamples"
                   +" -bg "+str(self.BackgroundFile)
                   +" -info "+str(self.AnnotationFile)
                   +" -w "+"40"
                   +" -h "+"40"
                   +" -num "+"532"
                   +" -vec "+str(self.defaultSourceDir)+'/gen/vec/pos.vec')
        print cmmd
        os.system(cmmd)
        print "done."

        """using opencv_haartraining to train classifier"""
        print "training classifier..."
        
        cmmd = str("opencv_traincascade"
                   +" -data "+str(self.defaultSourceDir)+'/gen'
                   +" -vec "+str(self.defaultSourceDir)+'/gen/vec/pos.vec'
                   +" -bg "+str(self.BackgroundFile)
                   +" -w "+"40"
                   +" -h "+"40"
                   +" -numPos "+"300"
                   +" -numNeg "+"700"
                   +" -numStages "+"7")
        print cmmd
        os.system(cmmd)
        print "done."

    def testDetector(self, _value=False):
        os.chdir(self.defaultImageDir)
        CD.anno_dir = self.defaultTestDir+'/Anno/'
        CD.test_dir = self.defaultTestDir+'/pos/'
        detector = CD.CascadeDetector("image")
        detector.test()
        # detector.detect_image(test_dir)
        filename_itor = CD.getItor(CD.anno_dir)
        for filename in filename_itor:
            print filename
            root = CD.getRoot(filename)
            detector.test_image(root)
        cv2.destroyAllWindows()

def main():
    
    app = QtGui.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()    