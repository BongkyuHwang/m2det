# coding: utf-8
def polygon_to_rbox(polygons):
    tl = polygons[:,0,:]
    tr = polygons[:,1,:]
    br = polygons[:,2,:]
    bl = polygons[:,3,:]
    dt, db = tr-tl, bl-br
    h = (np.linalg.norm(np.cross(dt, tl-br).reshape(-1,1), axis=1) + np.linalg.norm(np.cross(dt, tr-bl).reshape(-1,1), axis=1))/(2*np.linalg.norm(dt+eps, axis=1))
    p1 = (tl + bl) / 2.
    p2 = (tr + br) / 2.
    return np.hstack((p1, p2, h.reshape(-1,1)))
   
    
def polygon_to_rbox(polygons):
    tl = polygons[:,0,:]
    tr = polygons[:,1,:]
    br = polygons[:,2,:]
    bl = polygons[:,3,:]
    dt, db = tr-tl, bl-br
    h = (np.linalg.norm(np.cross(dt, tl-br).reshape(-1,1), axis=1) + np.linalg.norm(np.cross(dt, tr-bl).reshape(-1,1), axis=1))/(2*np.linalg.norm(dt+eps, axis=1))
    p1 = (tl + bl) / 2.
    p2 = (tr + br) / 2.
    return np.hstack((p1, p2, h.reshape(-1,1)))
   
    
