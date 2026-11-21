#!/usr/bin/env python3
"""Target-only namespace/mount sanity gate. Not a proof against hostile GPU code."""
import errno,json,os,socket
from pathlib import Path

def readonly(path):
    try:
        p=Path(path)/'.soup_boundary_write_test';p.write_text('must not be possible');p.unlink();return False
    except OSError as e:return e.errno in (errno.EROFS,errno.EACCES,errno.EPERM)

def main():
    status={k:v.strip() for k,v in (line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)}
    forbidden=('SOUP_QUEUE_WORKER_TOKEN','SOUP_QUEUE_REVIEWER_TOKEN','HF_TOKEN','GOOGLE_APPLICATION_CREDENTIALS','AWS_SECRET_ACCESS_KEY')
    checks={'nonroot_uid':os.geteuid()!=0,'effective_caps_zero':int(status['CapEff'],16)==0,'no_new_privileges':status.get('NoNewPrivs')=='1','only_loopback_network':set(os.listdir('/sys/class/net'))=={'lo'},'root_readonly':readonly('/'),'task_readonly':readonly('/task'),'cache_readonly':readonly('/hf_cache'),'runner_readonly':readonly('/runner'),'no_credential_environment':not any(k in os.environ for k in forbidden),'no_docker_socket':not Path('/var/run/docker.sock').exists(),'no_host_sentinel':not Path(os.environ['SOUP_HOST_SENTINEL']).exists()}
    output=Path('/output/selftest_write');output.write_text('own writable output');checks['output_writable']=output.read_text()=='own writable output'

    result={'status':'passed' if all(checks.values()) else 'failed','checks':checks,'uid':os.geteuid(),'gid':os.getegid(),'pid':os.getpid(),'hostname':socket.gethostname(),'not_hostile_code_security_proof':True}
    Path('/output/container_selftest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);raise SystemExit(result['status']!='passed')
if __name__=='__main__':main()
