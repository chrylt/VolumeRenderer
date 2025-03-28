// Random number generator (simple hash-based PCG Family)
uint hash(uvec3 v) {
    uint x = v.x * 73856093u;
    uint y = v.y * 19349663u;
    uint z = v.z * 83492791u;
    uint hash = x ^ y ^ z;
    hash = (hash ^ (hash >> 16u)) * 0x45d9f3b;
    hash = (hash ^ (hash >> 16u)) * 0x45d9f3b;
    hash = hash ^ (hash >> 16u);
    return hash;
}
float randf(uvec3 seed) {
    return float(hash(seed)) * (1.0 / 4294967295.0);
}

float randf_inc(inout uvec3 seed) {
    seed += uvec3(1, 1, 1);
    return randf(seed);
}

vec3 randomDir(float r1, float r2) {
    float theta = acos(1.0 - 2.0 * r1);
    float phi = 2.0 * M_PI * r2;
    return normalize(vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)));
}

bool intersectAABB(vec3 origin, vec3 dir, vec3 boxMin, vec3 boxMax, inout float tmin, inout float tmax) {
    for (int i = 0; i < 3; i++) {
        float invD = 1.0 / dir[i];
        float t0 = (boxMin[i] - origin[i]) * invD;
        float t1 = (boxMax[i] - origin[i]) * invD;
        if (invD < 0.0) {
            float temp = t0; t0 = t1; t1 = temp;
        }
        tmin = max(tmin, t0);
        tmax = min(tmax, t1);
        if (tmax < tmin) return false;
    }
    return true;
}

bool intersectSphere(in Ray ray, in vec3 sphereCenter, in float sphereRadius, out float t)
{
    vec3 oc = ray.origin - sphereCenter;
    float b = dot(oc, ray.direction);
    float c = dot(oc, oc) - sphereRadius * sphereRadius;
    float discriminant = b * b - c;

    if (discriminant < 0.0) {
        // No intersection
        return false;
    }

    float sqrtDisc = sqrt(discriminant);
    float t0 = -b - sqrtDisc;
    float t1 = -b + sqrtDisc;

    // We want the closest positive t
    if (t0 > 0.0) {
        t = t0;
    }
    else if (t1 > 0.0) {
        t = t1;
    }
    else {
        // Both intersections are behind the ray origin
        return false;
    }

    return true;
}

bool intersectSphere(in vec3 rayOrigin, in vec3 rayDir, in vec3 sphereCenter, in float sphereRadius, out float t)
{
    vec3 oc = rayOrigin - sphereCenter;
    float b = dot(oc, rayDir);
    float c = dot(oc, oc) - sphereRadius * sphereRadius;
    float discriminant = b * b - c;

    if (discriminant < 0.0) {
        // No intersection
        return false;
    }

    float sqrtDisc = sqrt(discriminant);
    float t0 = -b - sqrtDisc;
    float t1 = -b + sqrtDisc;

    // We want the closest positive t
    if (t0 > 0.0) {
        t = t0;
    }
    else if (t1 > 0.0) {
        t = t1;
    }
    else {
        // Both intersections are behind the ray origin
        return false;
    }

    return true;
}

bool intersectThickRay(
    vec3 cameraPos,
    vec3 cameraDir,
    vec3 lineOrigin,
    vec3 lineDir,
    float width,
    out float tHit)
{
    cameraDir = normalize(cameraDir);
    lineDir = normalize(lineDir);

    vec3 w0 = cameraPos - lineOrigin;

    float a = dot(cameraDir, cameraDir); // ~1.0 if normalized
    float b = dot(cameraDir, lineDir);
    float c = dot(lineDir, lineDir);     // ~1.0 if normalized
    float d = dot(cameraDir, w0);
    float e = dot(lineDir, w0);

    float denom = a * c - b * b;
    if (abs(denom) < 1e-6)
    {
        // Lines are parallel or nearly parallel
        vec3 proj = lineOrigin + e * lineDir;
        float distSq = dot(proj - cameraPos, proj - cameraPos);
        if (distSq <= width * width)
        {
            // If parallel and within width, intersection occurs right at the start (t=0).
            tHit = 0.0;
            return true;
        }
        return false;
    }

    float invDenom = 1.0 / denom;
    float t = (b * e - c * d) * invDenom;
    float s = (a * e - b * d) * invDenom;

    // Compute closest points
    vec3 closestOnCameraRay = cameraPos + cameraDir * t;
    vec3 closestOnLine = lineOrigin + lineDir * s;

    vec3 diff = closestOnLine - closestOnCameraRay;
    float distSqClosest = dot(diff, diff);

    // Check width and ensure forward direction (t ? 0)
    if (distSqClosest <= width * width && t >= 0.0 && s >= 0.0)
    {
        tHit = t;
        return true;
    }

    return false;
}

bool intersectPointLights(in Ray ray) {  // Visualize positions of point lights
    for (uint i = 0; i < lightCount; ++i) {
        RayLight light = lights[i];

        float t_discard;
        if (intersectSphere(ray, light.positionTo, 0.2, t_discard))
            return true;
    }
    return false;
}

bool intersectRayLights(in Ray ray) {  // Visualize positions of point lights
    for (uint i = 0; i < lightCount; ++i) {
        RayLight light = lights[i];

        float t_discard;    // TODO, direction vs end point not functional yet
        if (intersectThickRay(ray.origin, ray.direction, lights[i].positionFrom, lights[i].positionTo, 0.1, t_discard)) {
            return true;
        }
    }
    return false;
}

vec3 getClosestPointOnSphere(vec3 sphereOrigin, float radius, vec3 refPosition) {
    return sphereOrigin + normalize(refPosition - sphereOrigin) * radius;
}

vec3 getPointLightContribution(vec3 incomingPos, vec3 lightPos, float lightIntensity) {

    vec3 L = lightPos - incomingPos;
    float dist2 = dot(L, L);
    if (dist2 < 0.0001) return vec3(0); // guard against division by zero
    float dist = sqrt(dist2);
    float attenuation = lightIntensity / (4.0 * M_PI * dist2);
    return attenuation * vec3(1.0); // assuming white lights
}

vec3 getSphereLightContribution(vec3 incomingPos, vec3 sphereOrigin, float lightIntensity) {

    vec3 currPosition = getClosestPointOnSphere(sphereOrigin, beamRadius, incomingPos); // turning it into sphere light

    return getPointLightContribution(incomingPos, currPosition, lightIntensity);
}