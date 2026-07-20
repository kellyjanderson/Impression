from impression.modeling import Section, Station, loft_sections, as_section
from impression.modeling.drawing2d import make_rect


def build():
    left0 = make_rect(size=(1.0, 0.8), center=(-1.2, 0.0), color="#7ba4c2")
    right0 = make_rect(size=(0.9, 0.9), center=(1.2, 0.0), color="#d6a37c")
    left1 = make_rect(size=(0.8, 1.0), center=(-1.3, 0.0), color="#7ba4c2")
    right1 = make_rect(size=(1.1, 0.7), center=(1.3, 0.0), color="#d6a37c")

    section0 = Section((as_section(left0).regions[0], as_section(right0).regions[0]))
    section1 = Section((as_section(right1).regions[0], as_section(left1).regions[0]))

    stations = [
        Station(
            t=0.0,
            section=section0,
            origin=(0.0, 0.0, 0.0),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 1.0, 0.0),
            n=(0.0, 0.0, 1.0),
        ),
        Station(
            t=1.0,
            section=section1,
            origin=(0.2, 0.0, 1.2),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 1.0, 0.0),
            n=(0.0, 0.0, 1.0),
        ),
    ]
    return loft_sections(stations, samples=48, cap_ends=True)

