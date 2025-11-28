import numpy as np
import potrace
import cv2

def test_potrace():
    # Case 1: Small square in middle
    print("Testing Case 1: Small square in middle")
    data = np.zeros((100, 100), dtype=np.uint8)
    data[40:60, 40:60] = 1
    
    bmp = potrace.Bitmap(data)
    path = bmp.trace()
    
    print(f"Curves: {len(path.curves)}")
    for curve in path.curves:
        print(f"Start: {curve.start_point}")
        for segment in curve:
            print(f"Segment: {segment}")

    # Case 2: Inverted (Hole)
    print("\nTesting Case 2: Hole in middle")
    data2 = np.ones((100, 100), dtype=np.uint8)
    data2[40:60, 40:60] = 0
    
    bmp2 = potrace.Bitmap(data2)
    path2 = bmp2.trace()
    print(f"Curves: {len(path2.curves)}")
    
    # Case 3: Random noise
    print("\nTesting Case 3: Random noise")
    data3 = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
    # Case 4: uint32
    print("\nTesting Case 4: uint32")
    data4 = np.zeros((100, 100), dtype=np.uint32)
    data4[40:60, 40:60] = 1
    bmp4 = potrace.Bitmap(data4)
    path4 = bmp4.trace()
    print(f"Curves: {len(path4.curves)}")

    # Case 5: Inverted logic (0=black?)
    print("\nTesting Case 5: Inverted logic (0=black?)")
    data5 = np.ones((100, 100), dtype=np.uint8)
    data5[40:60, 40:60] = 0
    bmp5 = potrace.Bitmap(data5)
    path5 = bmp5.trace()
    # Case 6: Boolean
    print("\nTesting Case 6: Boolean")
    data6 = np.zeros((100, 100), dtype=bool)
    data6[40:60, 40:60] = True
    try:
        bmp6 = potrace.Bitmap(data6)
        path6 = bmp6.trace()
        print(f"Curves: {len(path6.curves)}")
    except Exception as e:
        print(f"Boolean failed: {e}")
    else:
        for i, curve in enumerate(path6.curves):
            print(f"Curve {i}: Start {curve.start_point}")
    # Case 7: Inverted Boolean
    print("\nTesting Case 7: Inverted Boolean")
    data7 = ~data6
    try:
        bmp7 = potrace.Bitmap(data7)
        path7 = bmp7.trace()
        print(f"Curves: {len(path7.curves)}")
        for i, curve in enumerate(path7.curves):
            print(f"Curve {i}: Start {curve.start_point}")
    except Exception as e:
        print(f"Inverted Boolean failed: {e}")





if __name__ == "__main__":
    test_potrace()
