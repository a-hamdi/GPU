#include "bfs.h"



// Function to generate a random large graph in CSR format
void generate_random_graph(int num_vertices, int* num_edges, int** edges, int** dest) {
    srand(time(NULL));
    
    // Allocate maximum possible space for destinations
    int max_edges = num_vertices * AVERAGE_EDGES_PER_VERTEX;
    *dest = (int*)malloc(max_edges * sizeof(int));
    *edges = (int*)malloc((num_vertices + 1) * sizeof(int));
    
    int current_edge = 0;
    (*edges)[0] = 0;
    
    // For each vertex
    for (int i = 0; i < num_vertices; i++) {
        // Generate random number of edges for this vertex
        int edges_for_vertex = rand() % (AVERAGE_EDGES_PER_VERTEX * 2);
        
        // Add random edges
        for (int j = 0; j < edges_for_vertex; j++) {
            int dest_vertex = rand() % num_vertices;
            if (dest_vertex != i) {  // Avoid self-loops
                (*dest)[current_edge++] = dest_vertex;
            }
        }
        
        (*edges)[i + 1] = current_edge;
    }
    
    *num_edges = current_edge;
    
    // Reallocate dest array to actual size
    *dest = (int*)realloc(*dest, current_edge * sizeof(int));
}



void cpu_bfs(int source, int num_vertices, int num_edges, int* edges, int* dest, int* label) {
    // Initialize labels
    for (int i = 0; i < num_vertices; i++) {
        label[i] = -1;
    }
    
    int* current_frontier = (int*)malloc(num_vertices * sizeof(int));
    int* next_frontier = (int*)malloc(num_vertices * sizeof(int));
    int current_size = 0;
    int next_size = 0;
    
    label[source] = 0;
    current_frontier[0] = source;
    current_size = 1;
    
    int level = 0;
    
    while (current_size > 0) {
        next_size = 0;
        level++;
        
        for (int i = 0; i < current_size; i++) {
            int vertex = current_frontier[i];
            
            for (int edge = edges[vertex]; edge < edges[vertex + 1]; edge++) {
                int neighbor = dest[edge];
                
                if (label[neighbor] == -1) {
                    label[neighbor] = level;
                    next_frontier[next_size++] = neighbor;
                }
            }
        }
        
        int* temp = current_frontier;
        current_frontier = next_frontier;
        next_frontier = temp;
        current_size = next_size;
    }
    
    free(current_frontier);
    free(next_frontier);
}
