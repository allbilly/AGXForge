/* Optional Mesa 26.2.4 userspace compiler adapter. SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "asahi/compiler/agx_compile.h"
#include "compiler/glsl_types.h"
#include "compiler/nir/nir_builder.h"

#define LIMIT 8192
static nir_def *defs[LIMIT];
static nir_variable *phis[LIMIT];

static _Noreturn void fail(const char *message)
{
   fprintf(stderr, "Mesa bridge: %s\n", message);
   exit(1);
}

static nir_def *value(unsigned id)
{
   if (id >= LIMIT || !defs[id]) fail("undefined value");
   return defs[id];
}

static nir_def *integer(nir_builder *b, unsigned id, unsigned bits)
{
   if (bits != 16 && bits != 32) fail("invalid integer width");
   return nir_u2uN(b, value(id), bits);
}

static nir_def *floating(nir_builder *b, unsigned id)
{
   return nir_f2f32(b, value(id));
}

static FILE *output(const char *directory, const char *name)
{
   char path[4096];
   if (snprintf(path, sizeof(path), "%s/%s", directory, name) >= sizeof(path))
      fail("output path too long");
   FILE *f = fopen(path, "wb");
   if (!f) fail("cannot open output");
   return f;
}

int main(int argc, char **argv)
{
   if (argc != 3) fail("usage: agxforge_mesa INPUT OUTPUT_DIRECTORY");
   FILE *input = fopen(argv[1], "r");
   if (!input) fail("cannot open input");
   unsigned bindings, threads;
   char line[256], extra;
   if (!fgets(line, sizeof(line), input) ||
       sscanf(line, "AGXFORGE_MESA_V2 %u %u %c", &bindings, &threads, &extra) != 2 ||
       bindings < 1 || bindings > 8 || threads < 1 || threads >= 0x80000000)
      fail("invalid header");

   glsl_type_singleton_init_or_ref();
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                 &agx_nir_options, "AGXForge Mesa");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   nir_def *pointers[8];
   for (unsigned i = 0; i < bindings; ++i)
      pointers[i] = nir_load_preamble(&b, 1, 64, .base = i * 4);
   nir_def *thread = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_push_if(&b, nir_ult_imm(&b, thread, threads));

   unsigned loop_depth = 0, instructions = 0;
   while (fgets(line, sizeof(line), input)) {
      char op[32];
      unsigned d, a, x, y, parameter;
      if (++instructions > LIMIT ||
          sscanf(line, "%31s %u %u %u %u %u %c", op, &d, &a, &x, &y, &parameter, &extra) != 6 ||
          d >= LIMIT) fail("invalid instruction");
      nir_def *result = NULL;
      if (!strcmp(op, "imm")) {
         if (parameter != 16 && parameter != 32) fail("invalid constant width");
         result = nir_imm_intN_t(&b, a, parameter);
      } else if (!strcmp(op, "thread")) {
         switch (parameter) {
         case 80: result = thread; break;
         case 48: result = nir_channel(&b, nir_load_local_invocation_id(&b), 0); break;
         case 0: result = nir_channel(&b, nir_load_workgroup_id(&b), 0); break;
         case 52: result = nir_load_subgroup_invocation(&b); break;
         case 53: result = nir_load_subgroup_id(&b); break;
         default: fail("unsupported builtin");
         }
      } else if (!strcmp(op, "load") || !strcmp(op, "store")) {
         if (a >= bindings || (parameter != 16 && parameter != 32))
            fail("invalid memory binding or width");
         nir_def *address = nir_iadd(&b, pointers[a],
            nir_imul_imm(&b, nir_u2u64(&b, value(x)), parameter / 8));
         if (!strcmp(op, "load"))
            result = nir_load_global(&b, 1, parameter, address,
                                     .align_mul = parameter / 8);
         else {
            if (value(y)->bit_size != parameter) fail("store width mismatch");
            nir_store_global(&b, value(y), address, .align_mul = parameter / 8);
         }
      } else if (!strcmp(op, "phi_seed")) {
         if (phis[d]) fail("duplicate phi");
         if (parameter != 16 && parameter != 32) fail("invalid phi width");
         phis[d] = nir_local_variable_create(b.impl,
            parameter == 16 ? glsl_uint16_t_type() : glsl_uint_type(), "carried");
         if (value(a)->bit_size != parameter) fail("phi seed width mismatch");
         nir_store_var(&b, phis[d], value(a), 1);
      } else if (!strcmp(op, "loop_begin")) {
         if (loop_depth++) fail("nested loops outside profile");
         nir_push_loop(&b);
      } else if (!strcmp(op, "phi_load")) {
         if (!loop_depth || !phis[d]) fail("phi outside loop");
         result = nir_load_var(&b, phis[d]);
      } else if (!strcmp(op, "phi_store")) {
         if (!loop_depth || !phis[d]) fail("phi outside loop");
         nir_store_var(&b, phis[d], value(a), 1);
      } else if (!strcmp(op, "loop_end")) {
         if (!loop_depth) fail("unbalanced loop");
         nir_push_if(&b, nir_inot(&b, value(a)));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);
         nir_pop_loop(&b, NULL);
         loop_depth--;
         for (unsigned j = 0; j < LIMIT; ++j)
            if (phis[j]) defs[j] = nir_load_var(&b, phis[j]);
      } else if (!strcmp(op, "cmp")) {
         result = nir_ult(&b, integer(&b, a, 32), integer(&b, x, 32));
         if (parameter) result = nir_iand(&b, result, nir_ult_imm(&b, integer(&b, a, 32), parameter));
      } else if (!strcmp(op, "icmp") || !strcmp(op, "fcmp")) {
         bool fp = !strcmp(op, "fcmp");
         nir_def *left = fp ? floating(&b, a) : integer(&b, a, 32);
         nir_def *right = fp ? floating(&b, x) : integer(&b, x, 32);
         nir_def *predicate;
         switch (parameter) {
         case 0: predicate = fp ? nir_feq(&b, left, right) : nir_ieq(&b, left, right); break;
         case 1: predicate = fp ? nir_flt(&b, left, right) : nir_ult(&b, left, right); break;
         case 2: predicate = fp ? nir_flt(&b, right, left) : nir_ult(&b, right, left); break;
         default: fail("unsupported comparison");
         }
         result = nir_b2i32(&b, predicate);
      } else if (!strcmp(op, "select")) {
         result = nir_bcsel(&b, nir_ine_imm(&b, value(a), 0),
                            integer(&b, x, parameter), integer(&b, y, parameter));
      } else {
#define FLOAT2(name, builder) if (!strcmp(op, name)) result = builder(&b, floating(&b, a), floating(&b, x)); else
#define INT2(name, builder) if (!strcmp(op, name)) result = builder(&b, integer(&b, a, parameter), integer(&b, x, parameter)); else
#define FLOAT1(name, builder) if (!strcmp(op, name)) result = builder(&b, floating(&b, a)); else
         FLOAT2("fadd", nir_fadd)
         FLOAT2("fsub", nir_fsub)
         FLOAT2("fmul", nir_fmul)
         INT2("add", nir_iadd)
         INT2("sub", nir_isub)
         INT2("mul", nir_imul)
         INT2("and", nir_iand)
         INT2("or", nir_ior)
         INT2("xor", nir_ixor)
         FLOAT1("rcp", nir_frcp)
         FLOAT1("rsqrt", nir_frsq)
         FLOAT1("exp2", nir_fexp2)
         FLOAT1("log2", nir_flog2)
         if (!strcmp(op, "shl") || !strcmp(op, "shr")) {
            /* G13 shifts zero-extend the source, use the low seven count
             * bits and narrow only the result. NIR instead masks the count
             * to its source width, so explicitly zero counts 32..127. */
            nir_def *source = integer(&b, a, 32);
            nir_def *count = nir_iand_imm(&b, integer(&b, x, 32), 0x7f);
            nir_def *shifted = !strcmp(op, "shl") ? nir_ishl(&b, source, count)
                                                   : nir_ushr(&b, source, count);
            result = nir_u2uN(&b, nir_bcsel(&b, nir_ult_imm(&b, count, 32),
                                          shifted, nir_imm_int(&b, 0)), parameter);
         }
         else if (!strcmp(op, "madd")) result = nir_iadd(&b,
            nir_imul(&b, integer(&b, a, parameter), integer(&b, x, parameter)), integer(&b, y, parameter));
         else if (!strcmp(op, "fma")) result = nir_ffma(&b, floating(&b, a), floating(&b, x), floating(&b, y));
         else if (!strcmp(op, "fmax") || !strcmp(op, "fmin")) {
            nir_def *left = floating(&b, a), *right = floating(&b, x);
            nir_def *predicate = !strcmp(op, "fmax") ? nir_flt(&b, right, left) : nir_flt(&b, left, right);
            result = nir_bcsel(&b, predicate, left, right);
         }
         else if (!strcmp(op, "sin_turns")) result = nir_fsin_agx(&b,
            nir_fmul_imm(&b, nir_ffract(&b, floating(&b, a)), 4.0));
         else if (!strcmp(op, "u32_to_f32")) result = nir_u2f32(&b, integer(&b, a, 32));
         else if (!strcmp(op, "i32_to_f32")) result = nir_i2f32(&b, integer(&b, a, 32));
         else if (!strcmp(op, "f32_to_u32")) result = nir_f2u32(&b, floating(&b, a));
         else if (!strcmp(op, "f32_to_i32")) result = nir_f2i32(&b, floating(&b, a));
         else if (!strcmp(op, "f16_to_f32")) result = nir_f2f32(&b, value(a));
         else if (!strcmp(op, "f32_to_f16_rte")) result = nir_f2f16_rtne(&b, value(a));
         else if (!strcmp(op, "bitcast")) result = value(a);
         else fail("unsupported operation");
         /* Arithmetic values carry an explicit IR destination width. */
         if (result->bit_size != parameter &&
             (!strcmp(op, "fadd") || !strcmp(op, "fsub") || !strcmp(op, "fmul") || !strcmp(op, "fma") ||
              !strcmp(op, "fmax") || !strcmp(op, "fmin") || !strcmp(op, "rcp") || !strcmp(op, "rsqrt") ||
              !strcmp(op, "exp2") || !strcmp(op, "log2") || !strcmp(op, "sin_turns")))
            result = nir_f2fN(&b, result, parameter);
      }
      if (result) {
         if (defs[d]) fail("duplicate definition");
         defs[d] = result;
      }
   }
   if (ferror(input) || loop_depth) fail("truncated input or unbalanced loop");
   fclose(input);
   nir_pop_if(&b, NULL);
   nir_validate_shader(b.shader, "AGXForge lowering");
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   agx_preprocess_nir(b.shader);
   FILE *f = output(argv[2], "shader.nir");
   nir_print_shader(b.shader, f);
   if (fclose(f)) fail("cannot write NIR");
   struct agx_shader_key key = {
      .dev = {.needs_g13x_coherency = U_TRISTATE_NO, .soft_fault = false},
      .reserved_preamble = bindings * 4,
      .secondary = true,
   };
   struct agx_shader_part part = {0};
   agx_compile_shader_nir(b.shader, &key, &part);
   struct agx_shader_info *i = &part.info;
   if (i->has_preamble || i->main_offset || i->scratch_size ||
       i->preamble_scratch_size || i->local_size || i->rodata.size_16 ||
       i->nr_gprs > 80 || i->push_count != bindings * 4 ||
       i->main_size != i->binary_size)
      fail("compiler output outside macOS carrier profile");
   f = output(argv[2], "shader.bin");
   if (fwrite(part.binary, 1, i->binary_size, f) != i->binary_size || fclose(f))
      fail("cannot write binary");
   f = output(argv[2], "metadata.json");
   fprintf(f, "{\"compiler\":\"Mesa 26.2.4 AGX\",\"register_halfs\":%u,"
      "\"uniform_halfs\":%u,\"binary_size\":%u,\"main_size\":%u,"
      "\"main_offset\":%u,\"has_preamble\":false,\"scratch_bytes\":0,"
      "\"threadgroup_bytes\":0,\"rodata_halfs\":0,\"workgroup_size\":[32,1,1]}\n",
      i->nr_gprs, i->push_count, i->binary_size, i->main_size, i->main_offset);
   if (fclose(f)) fail("cannot write metadata");
   free(part.binary);
   ralloc_free(b.shader);
   glsl_type_singleton_decref();
   return 0;
}
