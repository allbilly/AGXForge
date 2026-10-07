// Passive capture of the exact authored pair's Metal Submit. The original call
// runs; this script changes no arguments, bytes, or return values.
'use strict';
let armed=false,lastStorage=null,seq=0,allocations=[],allocationRequests=[];
function emit(kind,fields){send(Object.assign({kind,sequence:seq++},fields));}
function hex(p,n){if(p.isNull()||n<0||n>65536)throw new Error('invalid read');return Array.from(new Uint8Array(p.readByteArray(n)),x=>x.toString(16).padStart(2,'0')).join('');}
function snapshot(p){
 const out={storage:p.toString(),shmems:{}};
 for(const [name,offset] of [['kernel',0x20],['segment',0x60]]){
  const obj=p.add(offset).readPointer(),row={object:obj.toString()};
  if(!obj.isNull()){
   const cpu=obj.add(0x88).readPointer(),size=obj.add(0x84).readU32();
   row.id=obj.add(0x80).readU32();row.size=size;row.cpu=cpu.toString();
   if(!cpu.isNull()&&size<=16384)row.hex=hex(cpu,size);
  }
  out.shmems[name]=row;
 }
 return out;
}
function install(){
 if(armed)return;
 const m=Process.findModuleByName('IOGPU');if(!m)return;
 const submit=m.findExportByName('IOGPUCommandQueueSubmitCommandBuffers');
 const create=m.findExportByName('IOGPUMetalCommandBufferStoragePoolCreateStorage');
 const ready=Process.mainModule.findExportByName('g17_capture_ready');
 if(!submit||!create||!ready)throw new Error('required export absent');
 Interceptor.attach(create,{onLeave(r){if(!r.isNull())lastStorage=ptr(r.toString());}});
 const kit=Process.findModuleByName('IOKit');
 const call=kit&&kit.findExportByName('IOConnectCallMethod');
 if(!call)throw new Error('IOConnectCallMethod absent');
 Interceptor.attach(call,{
  onEnter(a){this.sel=a[1].toUInt32();if(this.sel===9){this.out=a[8];this.outSize=a[9];
    const n=a[5].toUInt32();this.input=n<=256&&!a[4].isNull()?hex(a[4],n):null;}},
  onLeave(r){if(this.sel!==9||r.toInt32()!==0)return;
   try{const n=Number(this.outSize.readU64().toString());if(n!==88)return;
    const words=[];for(let i=0;i<11;i++)words.push(this.out.add(i*8).readU64().toString());
    allocations.push(words);
    allocationRequests.push({input_hex:this.input,output_words:words});
   }catch(e){emit('capture_error',{where:'allocation',error:String(e)});}}
 });
 Interceptor.attach(submit,{onEnter(a){
  try{
   const q=a[0],count=a[2].toUInt32(),records=a[3],stride=a[4].toUInt32();
   const bytes=(count>0&&count<=4&&stride>=0x18&&stride<=0x100)?count*stride:0;
   const pointers=[];
   if(bytes>=0x20){for(const offset of [0x10,0x18]){
    const p=records.add(offset).readPointer(),region=Process.findRangeByAddress(p);
    const available=region&&!region.protection.startsWith('---')?
      Math.max(0,Math.min(128,Number(region.base.add(region.size).sub(p).toString()))):0;
    pointers.push({offset,pointer:p.toString(),region:region?{base:region.base.toString(),size:region.size,protection:region.protection}:null,
      hex:available?hex(p,available):null});
   }}
   const windows=[];
   for(const a of allocations){
    const aperture=Number(a[0]),host=ptr(a[1]),size=Number(a[5]);
    if(!host.isNull() && size<=0x20000){
      const pages=[];
      for(let offset=0;offset<size;offset+=4096){
       const data=new Uint8Array(host.add(offset).readByteArray(Math.min(4096,size-offset)));
       if(data.some(x=>x!==0))pages.push({offset,hex:Array.from(data,x=>x.toString(16).padStart(2,'0')).join('')});
      }
      windows.push({aperture:a[0],host:a[1],size,pages});
    }
   }
   emit('submit_enter',{queue:q.toString(),count,stride,records:records.toString(),
      record_hex:bytes?hex(records,bytes):null,pointers,allocations,allocationRequests,windows,
      snapshot:lastStorage?snapshot(lastStorage):null});
  }catch(e){emit('capture_error',{error:String(e)});}},
  onLeave(r){emit('submit_leave',{status:r.toInt32()});}});
 armed=true;
 ready.writeU32(0x17C0DE02);
 emit('instrumentation_ready',{image:m.path,submit:submit.toString()});
}
Interceptor.attach(Module.getGlobalExportByName('dlopen'),{onLeave(){try{install();}catch(e){emit('install_error',{error:String(e)});}}});
try{install();}catch(e){emit('install_error',{error:String(e)});}
