#ifndef ENGINE_METAL_H
#define ENGINE_METAL_H
#include "engine.h"
class Engine_Metal : public Engine {
public:
 static Engine_Metal* New(const Operator* op);
 virtual ~Engine_Metal();
 virtual void Init();
 virtual void Reset();
 virtual bool IterateTS(unsigned int count);
protected:
 Engine_Metal(const Operator* op);
private:
 struct Impl;
 Impl* impl;
 void Step(bool voltage);
};
#endif
