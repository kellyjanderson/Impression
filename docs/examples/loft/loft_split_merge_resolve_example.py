from impression.modeling import Section, Station, as_section, loft_sections, translate
from impression.modeling.drawing2d import PlanarShape2D, make_circle, make_rect


def _region_merge_demo():
    left = make_rect(size=(0.9, 0.9), center=(-0.2, 0.0), color="#8fb6d9")
    right = make_rect(size=(0.9, 0.9), center=(0.2, 0.0), color="#8fb6d9")
    merged = make_rect(size=(1.2, 1.0), center=(0.0, 0.0), color="#8fb6d9")
    s0 = Section((as_section(left).regions[0], as_section(right).regions[0]))
    s1 = Section((as_section(merged).regions[0],))
    stations = [
        Station(t=0.0, section=s0, origin=(0.0, 0.0, 0.0), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0)),
        Station(t=1.0, section=s1, origin=(0.0, 0.0, 2.0), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0)),
    ]
    return loft_sections(stations, samples=40, cap_ends=True, split_merge_mode="resolve", split_merge_steps=10)


def _hole_merge_demo():
    outer = make_rect(size=(2.0, 1.4)).outer
    hole_a = make_circle(radius=0.16, center=(-0.30, 0.0)).outer
    hole_b = make_circle(radius=0.16, center=(0.30, 0.0)).outer
    merged_hole = make_circle(radius=0.42, center=(0.0, 0.0)).outer
    start = PlanarShape2D(outer=outer, holes=[hole_a, hole_b], color="#a6c7b2")
    end = PlanarShape2D(outer=outer, holes=[merged_hole], color="#a6c7b2")
    stations = [
        Station(t=0.0, section=as_section(start), origin=(0.0, 0.0, 0.0), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0)),
        Station(t=1.0, section=as_section(end), origin=(0.0, 0.0, 2.0), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0)),
    ]
    return loft_sections(stations, samples=40, cap_ends=True, split_merge_mode="resolve", split_merge_steps=10)


def build():
    a = _region_merge_demo()
    b = _hole_merge_demo()
    translate(a, (-2.3, 0.0, 0.0))
    translate(b, (2.3, 0.0, 0.0))
    return [a, b]
