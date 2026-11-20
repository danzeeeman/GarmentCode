""" Panels for a straight upper garment (T-shirt)
    Note that the code is very similar to Bodice. 
"""
import numpy as np
import pygarment as pyg

from assets.garment_programs.base_classes import BaseBodicePanel


def _side_edges(b_width, width, length, body, design, body_waist):
    """Vertices of the side seam of a straight torso panel, hem -> armhole level.

        With `waist_shape` > 0 (jackets, coats) the seam is taken in at the waist: 1 brings the panel to
        the body's waist measurement plus ease. Only panels long enough to pass the waist are shaped.
    """
    hem, top = [-b_width, 0], [-width, length]
    shape = design.get('waist_shape', {}).get('v', 0.)
    y_waist = length - body['waist_line']
    y_bust = length - body['_bust_line']
    if shape <= 0 or y_waist < 5 or y_bust - y_waist < 5:
        return [hem, top]

    def on_line(y):
        return b_width + (width - b_width) * y / length

    take_in = shape * max(on_line(y_waist) - body_waist * 1.1, 0)
    if take_in < 0.5:
        return [hem, top]
    return [hem, [-(on_line(y_waist) - take_in), y_waist], [-on_line(y_bust), y_bust], top]


class TorsoFrontHalfPanel(BaseBodicePanel):
    """Half of a simple non-fitted upper garment (e.g. T-Shirt)
    
        Fits to the bust size
    """
    def __init__(self, name, body, design) -> None:
        """ Front = True, provides the adjustments necessary for the front panel
        """
        super().__init__(name, body, design)

        design = design['shirt']

        # width
        m_width = design['width']['v'] * body['bust']
        b_width = design['flare']['v'] * m_width

        # sizes 
        body_width = (body['bust'] - body['back_width']) / 2 
        frac = body_width / body['bust'] 
        self.width = frac * m_width
        b_width = frac * b_width

        sh_tan = np.tan(np.deg2rad(body['_shoulder_incl']))
        shoulder_incl = sh_tan * self.width
        length = design['length']['v'] * body['waist_line']

        # length in the front panel is adjusted due to shoulder inclination
        # for the correct sleeve fitting
        fb_diff = (frac - (0.5 - frac)) * body['bust']
        length = length - sh_tan * fb_diff

        side = _side_edges(
            b_width, self.width, length, body, design, (body['waist'] - body['waist_back_width']) / 2)
        self.edges = pyg.EdgeSeqFactory.from_verts(
            [0, 0], 
            *side, 
            [0, length + shoulder_incl], 
            loop=True
        )

        # Interfaces
        self.interfaces = {
            'outside':  pyg.Interface(self, self.edges[1:-2]),   
            'inside': pyg.Interface(self, self.edges[-1]),
            'shoulder': pyg.Interface(self, self.edges[-2]),
            'bottom': pyg.Interface(self, self.edges[0], ruffle=self.edges[0].length() / ((body['waist'] - body['waist_back_width']) / 2)),
            
            # Reference to the corner for sleeve and collar projections
            'shoulder_corner': pyg.Interface(self, [self.edges[-3], self.edges[-2]]),
            'collar_corner': pyg.Interface(self, [self.edges[-2], self.edges[-1]])
        }

        # default placement
        self.translate_by([0, body['height'] - body['head_l'] - length - shoulder_incl, 0])

    def get_width(self, level):
        return super().get_width(level) + self.width - self.body['shoulder_w'] / 2


class TorsoBackHalfPanel(BaseBodicePanel):
    """Half of a simple non-fitted upper garment (e.g. T-Shirt)
    
        Fits to the bust size
    """
    def __init__(self, name, body, design) -> None:
        """ Front = True, provides the adjustments necessary for the front panel
        """
        super().__init__(name, body, design)

        design = design['shirt']
        # account for ease in basic measurements
        m_width = design['width']['v'] * body['bust']
        b_width = design['flare']['v'] * m_width

        # sizes 
        body_width = body['back_width'] / 2
        frac = body_width / body['bust'] 
        self.width = frac * m_width
        b_width = frac * b_width

        shoulder_incl = (np.tan(np.deg2rad(body['_shoulder_incl']))) * self.width
        length = design['length']['v'] * body['waist_line']

        side = _side_edges(b_width, self.width, length, body, design, body['waist_back_width'] / 2)
        self.edges = pyg.EdgeSeqFactory.from_verts(
            [0, 0], 
            *side, 
            [0, length + shoulder_incl], 
            loop=True
        )

        # Interfaces
        self.interfaces = {
            'outside':  pyg.Interface(self, self.edges[1:-2]),   
            'inside': pyg.Interface(self, self.edges[-1]),
            'shoulder': pyg.Interface(self, self.edges[-2]),
            'bottom': pyg.Interface(self, self.edges[0], ruffle=self.edges[0].length() / (body['waist_back_width'] / 2)),
            
            # Reference to the corner for sleeve and collar projections
            'shoulder_corner': pyg.Interface(self, [self.edges[-3], self.edges[-2]]),
            'collar_corner': pyg.Interface(self, [self.edges[-2], self.edges[-1]])
        }

        # default placement
        self.translate_by([0, body['height'] - body['head_l'] - length - shoulder_incl, 0])

    def get_width(self, level):
        return super().get_width(level) + self.width - self.body['shoulder_w'] / 2
