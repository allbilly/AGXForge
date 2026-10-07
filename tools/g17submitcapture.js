// Passive capture for the explicitly authorized Metal reference process only.
// No replacements, argument writes, extra submissions, or callback invocations.
'use strict';
let sequence = 0;
let attached = false;
function emit(kind, fields) { send(Object.assign({kind, sequence: sequence++, thread: Process.getCurrentThreadId()}, fields)); }
function bytes(p,n) {
  if(n===0) return '';
  if(p.isNull() || n<0 || n>4096) throw new Error('invalid bounded read');
  return Array.from(new Uint8Array(p.readByteArray(n)), x=>x.toString(16).padStart(2,'0')).join('');
}
function recordError(where,e) { emit('capture_error',{where,error:String(e)}); }
function hook(module,name,callbacks) {
  const address=module.findExportByName(name);
  if(!address)throw new Error('missing export '+name);
  Interceptor.attach(address,callbacks);
}
function storage(p) {
  const result={pointer:p.toString(),trace_id:p.add(0x320).readU64().toString(),shmems:{}};
  for(const [name,offset] of [['kernel',0x20],['segment',0x60],['sideband',0x40],['debug',0x368]]) {
    const object=p.add(offset).readPointer();
    result.shmems[name]={object:object.toString()};
    if(!object.isNull())Object.assign(result.shmems[name],{
      id:object.add(0x80).readU32(),size:object.add(0x84).readU32(),cpu_address:object.add(0x88).readPointer().toString()});
  }
  return result;
}
function install(iogpu) {
  if(attached)return;
  // Bound offsets to the observed code before reading private storage layouts.
  const sub=iogpu.findExportByName('IOGPUCommandQueueSubmitCommandBuffers');
  if(!sub || bytes(sub.add(0x74),4)!=='a80a40f9')throw new Error('Submit layout differs from 25G83 snapshot');
  attached=true;
  emit('instrumentation_ready',{image:iogpu.path,base:iogpu.base.toString(),submit:sub.toString()});
  hook(iogpu,'IOGPUCommandQueueSubmitCommandBuffers',{
    onEnter(a){try{
      this.id=sequence;this.out=a[5];
      const count=a[2].toUInt32(),stride=a[4].toUInt32();
      if(count<1 || count>16 || stride<48 || stride>256 || count*stride>4096)throw new Error('unexpected submission dimensions');
      emit('submit_enter',{id:this.id,queue:a[0].toString(),flags:a[1].toString(),count,stride,input:a[3].toString(),hex:bytes(a[3],count*stride)});
    }catch(e){recordError('submit_enter',e);}},
    onLeave(r){try{emit('submit_leave',{id:this.id,status:r.toInt32(),output_word:this.out&&!this.out.isNull()?this.out.readU32():null});}catch(e){recordError('submit_leave',e);}}
  });
  hook(iogpu,'IOGPUMetalCommandBufferStoragePoolCreateStorage',{
    onEnter(a){this.trace=a[1].toString();},
    onLeave(r){try{if(!r.isNull())emit('storage_created',{argument_trace_id:this.trace,storage:storage(r)});}catch(e){recordError('storage_created',e);}}
  });
  hook(iogpu,'IOGPUMetalCommandBufferStorageFinalizeShmemHeader',{
    onEnter(a){this.p=a[0];},
    onLeave(){try{emit('storage_finalized',{storage:storage(this.p)});}catch(e){recordError('storage_finalized',e);}}
  });
  const kit=Process.getModuleByName('IOKit');
  hook(kit,'IOConnectTrap4',{
    onEnter(a){this.port=a[0].toString();this.index=a[1].toUInt32();emit('trap4_enter',{port:this.port,index:this.index,arg1:a[2].toString(),arg2:a[3].toString(),arg3:a[4].toString(),arg4:a[5].toString()});},
    onLeave(r){emit('trap4_leave',{port:this.port,index:this.index,status:r.toInt32()});}
  });
  hook(kit,'IOConnectCallMethod',{
    onEnter(a){this.selector=a[1].toUInt32();this.port=a[0].toString();this.out=a[8];this.outsize=a[9];},
    onLeave(r){if(![7,9,14,16,28,29,261].includes(this.selector))return;
      try{const n=this.outsize.isNull()?0:Number(this.outsize.readU64().toString());
        emit('iokit_method_leave',{port:this.port,selector:this.selector,status:r.toInt32(),output_bytes:n,
          output_hex:r.toInt32()===0&&n<=4096?bytes(this.out,n):null});
      }catch(e){recordError('iokit_method_leave',e);}}
  });
  hook(kit,'IODataQueueDequeue',{
    onEnter(a){this.data=a[1];this.size=a[2];},
    onLeave(r){if(r.toInt32()!==0)return;
      try{const n=this.size.readU32();emit('notification_dequeued',{bytes:n,hex:bytes(this.data,n)});}catch(e){recordError('notification_dequeued',e);}}
  });
}
// Resolve exports after dlopen returns, not in dyld's module-added callback:
// the latter re-entered ObjC initialization and crashed the CPU-only smoke probe.
function installIfLoaded() {
  if(attached)return;
  const m=Process.findModuleByName('IOGPU');
  if(m){try{install(m);}catch(e){recordError('install',e);}}
}
Interceptor.attach(Module.getGlobalExportByName('dlopen'),{onLeave(){installIfLoaded();}});
installIfLoaded();
