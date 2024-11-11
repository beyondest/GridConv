# Input shape torch.Size([1000, 2, 5, 5]) for handcrafted grid
# Output shape torch.Size([1000, 3, 5, 5]) for handcrafted grid
# Input shape torch.Size([1000, 17, 2]) for autogrid
# Output shape torch.Size([1000, 17, 3]) for autogrid

# test_custom_2d_unnorm.pth.tar
- key:('S9', 'Greeting', 'Greeting 1.60457274')
- value: np.ndarray(shape=(2711, 2x17), dtype=float32)

# test_custom_3d_unnorm.pth.tar
- key:('S9', 'Greeting', 'Greeting 1.60457274')
- value: dict
  - 'pelvis': np.ndarray(shape=(2711,3), dtype=float32)
  - 'joint_3d': np.ndarray(shape=(2711,3x17), dtype=float32)
  - 'camera': dict{fx,fy,cx,cy}

# Dataset Preprocessing (Before Training)
p2d = (p2d - 500 )/500
p3d = p3d/1000 

```before transpose
array([[[[ 8. ,  8. ],
         [ 8. ,  8. ],
         [ 8. ,  8. ],
         [ 8. ,  8. ],
         [ 8. ,  8. ]],

        [[ 1. ,  1. ],
         [ 9. ,  9. ],
         [ 9. ,  9. ],
         [ 9. ,  9. ],
         [ 1. ,  1. ]],

        [[ 2. ,  2. ],
         [15. , 15. ],
         [ 9.5,  9.5],
         [12. , 12. ],
         [ 5. ,  5. ]],

        [[ 3. ,  3. ],
         [16. , 16. ],
         [10. , 10. ],
         [13. , 13. ],
         [ 6. ,  6. ]],

        [[ 4. ,  4. ],
         [17. , 17. ],
         [11. , 11. ],
         [14. , 14. ],
         [ 7. ,  7. ]]]])
         ```
```final SGT output, Batch x 2 x 5 x 5 for pos2d, batch is 1
array([[[[ 8. ,  8. ,  8. ,  8. ,  8. ],
         [ 1. ,  9. ,  9. ,  9. ,  1. ],
         [ 2. , 15. ,  9.5, 12. ,  5. ],
         [ 3. , 16. , 10. , 13. ,  6. ],
         [ 4. , 17. , 11. , 14. ,  7. ]],

        [[ 8. ,  8. ,  8. ,  8. ,  8. ],
         [ 1. ,  9. ,  9. ,  9. ,  1. ],
         [ 2. , 15. ,  9.5, 12. ,  5. ],
         [ 3. , 16. , 10. , 13. ,  6. ],
         [ 4. , 17. , 11. , 14. ,  7. ]]]])
         ```