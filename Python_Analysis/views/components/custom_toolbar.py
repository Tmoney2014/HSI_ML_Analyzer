from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

class CustomToolbar(NavigationToolbar2QT):
    # Only keep Save button
    toolitems = [
        ('Save', 'Save the figure', 'filesave', 'save_figure'),
    ]

    def __init__(self, canvas, parent, coordinates=True):
        super().__init__(canvas, parent, coordinates)
        # Make coordinate label bold
        if hasattr(self, 'locLabel'):
            font = self.locLabel.font()
            font.setBold(True)
            font.setPointSize(10)
            self.locLabel.setFont(font)
