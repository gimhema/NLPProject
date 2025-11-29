#include "chat_classifier.h"
#include <iostream>

int main()
{
    ChatClassifier classifier;
    if (!classifier.initialize()) {
        std::cerr << "Failed to initialize ChatClassifier\n";
        return 1;
    }

//    std::string chat = "Example";
    float confidence = 0.0f;

    std::string category = classifier.classify(chat, &confidence);

    std::cout << "Chat: " << chat << "\n";
    std::cout << "Predicted category: " << category
              << " (confidence=" << confidence << ")\n";

    return 0;
}
