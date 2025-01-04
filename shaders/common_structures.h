
#define M_PI       3.14159265358979323846

struct RayLight {
    vec3 positionFrom;
    vec3 positionTo;
    float intensity;
};

struct Ray {
    vec3 origin;
    vec3 direction;
    float minDistance;
    float maxDistance;
};