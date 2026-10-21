"""Local-only IPC and subprocess lifecycle. No network command server."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def atomic_write(path, content):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(content)
    os.replace(temporary,path)


class Session:
    def __init__(self, build, directory, interface, sim=False, lock_fd=None):
        self.build, self.directory = Path(build).resolve(), Path(directory).resolve()
        self.interface, self.sim = interface, sim
        self.process = None
        self.lock_fd = lock_fd
        self.seq = 0
        self.pending = False
        self.stopped = False
        self.status = None
        self.log = None
        self.last_status_mtime = 0
        self.last_status_time = time.monotonic()

    def arm(self):
        if self.process or self.stopped:
            raise RuntimeError('This session cannot be rearmed. Start a fresh app session.')
        if self.sim != (self.interface == 'lo'):
            raise ValueError('Simulation requires lo; hardware requires an explicit non-loopback interface')
        if self.interface not in {name for _,name in socket.if_nameindex()}:
            raise ValueError('Unknown Ethernet interface')
        binary = self.build/'target/release/g1_deploy_onnx_ref'
        manifest = json.loads((self.build/'binary-manifest.json').read_text())
        if hashlib.sha256(binary.read_bytes()).hexdigest()!=manifest['sha256']:
            raise ValueError('Guarded binary hash mismatch; rebuild')
        if (self.build/'GUARDED_BUILD').read_text().strip()!='sonic-laptop-v2':
            raise ValueError('Not a guarded build')
        args = [str(binary),self.interface,'policy/release/model_decoder.onnx',str(self.directory/'motions'),
                '--obs-config','policy/release/observation_config.yaml',
                '--encoder-file','policy/release/model_encoder.onnx',
                '--planner-file','planner/target_vel/V2/planner_sonic.onnx',
                '--input-type','keyboard','--output-type','zmq']
        if self.sim: args.append('--disable-crc-check')
        self.heartbeat()
        env = os.environ.copy(); env['SONIC_GUARD_DIR'] = str(self.directory)
        env.pop('SONIC_GUARD_SIM',None)
        if self.sim: env['SONIC_GUARD_SIM']='1'
        self.log = (self.directory/'controller.log').open('w')
        # No terminal input can bypass the app. Keep the process alive on GUI failure
        # so its independent writer can latch damping when the lease expires.
        self.process = subprocess.Popen(args,cwd=self.build,env=env,stdin=subprocess.DEVNULL,
                                        stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True,
                                        pass_fds=(() if self.lock_fd is None else (self.lock_fd,)))
        (self.directory/'pid').write_text(str(self.process.pid))

    def heartbeat(self):
        atomic_write(self.directory/'lease',f'{time.monotonic():.9f} {int(not self.stopped)}\n')

    def stop(self):
        self.stopped=True
        self.heartbeat()

    def poll(self):
        path=self.directory/'status'
        try:
            lines=path.read_text().splitlines()
            seq,state,index,frame,playing,planner,fault,setup_ready=map(int,lines[0].split())
            result=dict(seq=seq,state=state,index=index,frame=frame,playing=bool(playing),
                        setup_ready=bool(setup_ready),planner=bool(planner),fault=bool(fault),result=lines[1],reason=lines[2] if len(lines)>2 else '')
            stamp=path.stat().st_mtime_ns
            if stamp!=self.last_status_mtime:
                self.last_status_mtime=stamp; self.last_status_time=time.monotonic()
            self.status=result
            if seq>=self.seq: self.pending=False
            if fault: self.stopped=True
        except (FileNotFoundError,ValueError,IndexError):
            pass
        if self.status and time.monotonic()-self.last_status_time>1:
            self.stop()  # Input/status loop stalled, even if UI is still responsive.
        return self.status

    def command(self, command, argument=0):
        if self.stopped or not self.process or self.process.poll() is not None:
            raise RuntimeError('Controller is not running or stop is latched')
        if self.pending and command!='idle': return False
        if command not in {'grounded_start','motion','planner','idle','move'}:
            raise ValueError('Unsupported command')
        self.seq+=1; self.pending=True
        atomic_write(self.directory/'request',f'{self.seq} {command} {int(argument)} {time.monotonic():.9f}\n')
        return True

    def terminate_supported(self):
        # Only the explicit "robot supported" UI action calls this. Stopping the
        # app alone must not terminate the damping publisher.
        self.stop()
        if self.process and self.process.poll() is None:
            self.process.terminate()
        if self.log: self.log.close()
