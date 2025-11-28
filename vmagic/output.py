import svgwrite

class SVGWriter:
    def __init__(self):
        pass

    def save(self, paths, output_path, size):
        """
        Saves paths to an SVG file.
        paths: list of {'path': <potrace path>, 'color': (r, g, b)}
        size: (height, width)
        """
        height, width = size
        dwg = svgwrite.Drawing(output_path, size=(width, height), profile='tiny')

        for item in paths:
            path_obj = item['path']
            color = item['color']
            rgb_str = f"rgb({color[0]},{color[1]},{color[2]})"
            
            # Convert potrace path to SVG path data
            # Potrace path consists of curves.
            # We need to iterate over curves and generate 'd' attribute.
            
            for curve in path_obj:
                d = []
                start = curve.start_point
                d.append(f"M {start.x} {start.y}")
                
                for segment in curve:
                    if segment.is_corner:
                        c = segment.c
                        end = segment.end_point
                        d.append(f"L {c.x} {c.y} L {end.x} {end.y}")
                    else:
                        c1 = segment.c1
                        c2 = segment.c2
                        end = segment.end_point
                        d.append(f"C {c1.x} {c1.y} {c2.x} {c2.y} {end.x} {end.y}")
                
                d.append("Z") # Close path
                
                dwg.add(dwg.path(d=" ".join(d), fill=rgb_str, stroke="none"))

        dwg.save()
