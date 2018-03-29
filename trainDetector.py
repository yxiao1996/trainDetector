import os
import sys
from PyQt4 import QtGui
from libs.ustr import ustr
import libs.annoCvt as annoCvt
import libs.bgGen as bgGen

__appname__ = 'trainDetector'

class Example(QtGui.QMainWindow):
    
    def __init__(self):
        super(Example, self).__init__()

        self.defaultSaveDir = '.'
        self.defaultImageDir = '.'
        self.filePath = '.'
        self.initUI()
        self.AnnotationFile = None
        self.BackgroundFile = None
        
    def initUI(self):               
        
        # Actions
        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)        
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.qApp.quit)

        setdirAction = QtGui.QAction('&Change save dir', self)
        setdirAction.triggered.connect(self.changeSavedirDialog)

        setimagedirAction = QtGui.QAction('&Change image dir', self)
        setdirAction.triggered.connect(self.changeImagedirDialog)

        readannoAction = QtGui.QAction('&Read Annotation', self)
        readannoAction.triggered.connect(self.openAnnotationDialog)

        readbgAction = QtGui.QAction('&Read Background', self)
        readbgAction.triggered.connect(self.openBackgroundDialog)

        # Buttons
        setimagedirButton = QtGui.QPushButton('image dir', self)
        setimagedirButton.clicked.connect(self.changeImagedirDialog)
        setimagedirButton.resize(setimagedirButton.sizeHint())
        setimagedirButton.move(50, 50)

        genfilesButton = QtGui.QPushButton('generate files', self)
        genfilesButton.clicked.connect(self.genFiles)
        genfilesButton.resize(genfilesButton.sizeHint())
        genfilesButton.move(50, 100)

        trainButton = QtGui.QPushButton('train detector', self)
        trainButton.clicked.connect(self.trainDetector)
        trainButton.resize(trainButton.sizeHint())
        trainButton.move(50, 150)

        # menuBar
        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)
        fileMenu.addAction(setdirAction)
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
                                                       '%s - Save detector to the directory' % __appname__, path,  QtGui.QFileDialog.ShowDirsOnly
                                                       | QtGui.QFileDialog.DontResolveSymlinks))
        if dirpath is not None and len(dirpath) > 1:
            self.defaultImageDir = dirpath

        self.statusBar().showMessage('%s . Use the images in  %s' %
                                     ('Change image folder', self.defaultImageDir))
        self.statusBar().show()

        self.AnnotationFile = str(self.defaultImageDir+'/info.dat')
        self.BackgroundFile = str(self.defaultImageDir+'/bg.txt')

    def changeSavedirDialog(self, _value=False):
        if self.defaultSaveDir is not None:
            path = ustr(self.defaultSaveDir)
        else:
            path = '.'

        dirpath = ustr(QtGui.QFileDialog.getExistingDirectory(self,
                                                       '%s - Save detector to the directory' % __appname__, path,  QtGui.QFileDialog.ShowDirsOnly
                                                       | QtGui.QFileDialog.DontResolveSymlinks))

        if dirpath is not None and len(dirpath) > 1:
            self.defaultSaveDir = dirpath

        self.statusBar().showMessage('%s . Detector will be saved to %s' %
                                     ('Change saved folder', self.defaultSaveDir))
        self.statusBar().show()

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
                                     ('Generate training files', self.defaultSaveDir))
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
                                     ('Generate training files', self.defaultSaveDir))
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
                   +" -vec "+str(self.defaultSaveDir)+'/gen/vec/pos.vec')
        print cmmd
        os.system(cmmd)
        print "done."

        """using opencv_haartraining to train classifier"""
        print "training classifier..."
        
        cmmd = str("opencv_traincascade"
                   +" -data "+str(self.defaultSaveDir)+'/gen'
                   +" -vec "+str(self.defaultSaveDir)+'/gen/vec/pos.vec'
                   +" -bg "+str(self.BackgroundFile)
                   +" -w "+"40"
                   +" -h "+"40"
                   +" -numPos "+"300"
                   +" -numNeg "+"700"
                   +" -numStages "+"7")
        print cmmd
        os.system(cmmd)
        print "done."

def main():
    
    app = QtGui.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()    