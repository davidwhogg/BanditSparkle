
import sys
from PySide import QtCode, QtGui



class SimulatorTab(QtGui.QWidget):


    def __init__(self, parent):

        super(SimulatorTab, self).__init__(parent)
        self.parent = parent

        sp = QtGui.QSizePolicy(
            QtGui.QSizePolicy.MinimumExpanding,
            QtGui.QSizePolicy.MinimumExpanding)
        sp.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sp)

        # Create a top-level horizontal layout to contain a MPL figure and
        # a vertical layout of settings..
        tab_layout = QtGui.QHBoxLayout(self)
        tab_layout.setContentsMargins(20, 20, 20, 20)

        # Create the left hand pane.
        summary_widget = QtGui.QWidget()
        summary_layout = QtGui.QVBoxLayout(summary_widget)

        # Notes.
        self.summary_notes = QtGui.QPlainTextEdit(self)
        summary_layout.addWidget(self.summary_notes)

        # External sources of information.
        hbox = QtGui.QHBoxLayout()

        # - Simbad
        self.btn_query_simbad = QtGui.QPushButton(self)
        self.btn_query_simbad.setText("Query Simbad..")
        self.btn_query_simbad.clicked.connect(self.query_simbad)

        hbox.addWidget(self.btn_query_simbad)
        hbox.addItem(QtGui.QSpacerItem(
            40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum))
        summary_layout.addLayout(hbox)
        tab_layout.addWidget(summary_widget)


        # Create a matplotlib widget in the right hand pane.


        self.figure = mpl.MPLWidget(None, tight_layout=True)
        sp = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sp.setHorizontalStretch(0)
        sp.setVerticalStretch(0)
        sp.setHeightForWidth(self.figure.sizePolicy().hasHeightForWidth())
        self.figure.setSizePolicy(sp)
        self.figure.setFixedWidth(400)
        tab_layout.addWidget(self.figure)

        self.ax_top_comparison = self.figure.figure.add_subplot(211)
        self.ax_bottom_comparison = self.figure.figure.add_subplot(212)

        return None


class AsteroseismicSimulatorWindow(QtGui.QMainWindow):

    def __init__(self, **kwargs):
        super(AsteroseismicSimulator, self).__init__()

        self.setObjectName("BanditSparkle")
        self.resize(1200, 600)

        self.move(QtGui.QApplication.desktop().screen().rect().center() \
            - self.rect().center())

        self.__init_ui__()

        return None


    def __init_ui__(self):

        cw = QtGui.QWidget(self)
        cw_vbox = QtGui.QVBoxLayout(cw)
        cw_vbox.setContentsMargins(10, 10, 10, 10)

        # Create two tabs.
        self.tabs = QtGui.QTabDidget(cw)
        self.tabs.setTabPosition(QtGui.QTabWidget.North)
        self.tabs.setUsesScrollButtons(False)

        sp = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.MinimumExpanding)
        sp.setHorizontalStretch(0)
        sp.setVerticalStretch(0)
        sp.setHeightForWidth(self.tabs.sizePolicy().hasHeightForWidth())
        self.tabs.setSizePolicy(sp)

        # Simulator tab
        self.simulator_tab = SimulatorTab

        # Log-likelihood explorer

        cw_vbox.addWidget(self.tabs)
        self.setCentralWidget(cw)
        self.tabs.setCurrentIndex(0)

        return None




if __name__ == "__main__":

    try:
        app = QtGui.QApplication(sys.argv)

    except RuntimeError:
        None


    app.window = AsteroseismicSimulatorWindow()
    app.window.show()
    app.window.raise_()
    sys.exit(app.exec_())
