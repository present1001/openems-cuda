#include "operator_metal.h"
#include "engine_metal.h"
Operator_Metal* Operator_Metal::New() {
 Operator_Metal* op = new Operator_Metal(); op->Init(); return op;
}
Engine* Operator_Metal::CreateEngine() {
 m_Engine = Engine_Metal::New(this);return m_Engine;
}
