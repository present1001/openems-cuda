#ifndef OPERATOR_METAL_H
#define OPERATOR_METAL_H
#include "operator.h"
class Operator_Metal : public Operator {
public:
 static Operator_Metal* New();
 virtual Engine* CreateEngine();
protected:
 Operator_Metal():Operator() {}
};
#endif
