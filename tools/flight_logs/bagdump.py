#!/usr/bin/env python3
"""rosbag2 db3에서 mavros 토픽을 CDR 직접 파싱해 CSV로 뽑는다 (ROS 없이 동작)."""
import sqlite3, struct, sys, math

class R:
    def __init__(self, buf):
        self.b = buf
        self.o = 4  # encapsulation header
    def align(self, n):
        rel = self.o - 4
        pad = (-rel) % n
        self.o += pad
    def u32(self):
        self.align(4); v = struct.unpack_from('<I', self.b, self.o)[0]; self.o += 4; return v
    def i32(self):
        self.align(4); v = struct.unpack_from('<i', self.b, self.o)[0]; self.o += 4; return v
    def u8(self):
        v = self.b[self.o]; self.o += 1; return v
    def f64(self):
        self.align(8); v = struct.unpack_from('<d', self.b, self.o)[0]; self.o += 8; return v
    def string(self):
        n = self.u32(); s = self.b[self.o:self.o+n-1].decode('utf-8', 'replace'); self.o += n; return s
    def header(self):
        sec = self.i32(); nsec = self.u32(); fid = self.string()
        return sec + nsec * 1e-9, fid

def quat_to_yaw(x, y, z, w):
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

def read(db, topic, parser):
    c = sqlite3.connect(db)
    tid = c.execute('select id from topics where name=?', (topic,)).fetchone()
    if not tid:
        return []
    out = []
    for t, data in c.execute('select timestamp, data from messages where topic_id=? order by timestamp', (tid[0],)):
        out.append((t * 1e-9,) + parser(R(data)))
    return out

def p_pose(r):
    ts, _ = r.header()
    x, y, z = r.f64(), r.f64(), r.f64()
    qx, qy, qz, qw = r.f64(), r.f64(), r.f64(), r.f64()
    return (ts, x, y, z, math.degrees(quat_to_yaw(qx, qy, qz, qw)))

def p_twist(r):
    ts, _ = r.header()
    return (ts, r.f64(), r.f64(), r.f64(), r.f64(), r.f64(), r.f64())

def p_ext(r):
    ts, _ = r.header()
    return (ts, r.u8(), r.u8())

def p_state(r):
    ts, _ = r.header()
    conn, armed, guided, manual = r.u8(), r.u8(), r.u8(), r.u8()
    mode = r.string()
    return (ts, armed, mode)

if __name__ == '__main__':
    db = sys.argv[1]
    what = sys.argv[2]
    tbl = {
        'pose':   ('/mavros/local_position/pose', p_pose),
        'vel':    ('/mavros/local_position/velocity_local', p_twist),
        'ext':    ('/mavros/extended_state', p_ext),
        'state':  ('/mavros/state', p_state),
        'sp_pos': ('/mavros/setpoint_position/local', p_pose),
        'sp_vel': ('/mavros/setpoint_velocity/cmd_vel', p_twist),
    }[what]
    for row in read(db, tbl[0], tbl[1]):
        print(','.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in row))
