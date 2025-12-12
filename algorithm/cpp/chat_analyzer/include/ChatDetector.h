#ifndef _CHAT_ML_CLASSIFIER_H
#define _CHAT_ML_CLASSIFIER_H

#include <vector>
#include <string>
#include <map>
#include <algorithm>

class ChatDetector
{
private:

	std::vector<std::string> m_classes;               // ["Car", "Electronics", ...]
	std::vector< std::vector<double> > m_coef;        // coef[c][feature]
	std::vector<double> m_intercept;                  // intercept[c]
	std::map<std::string, int> m_vocab;               // token → index (Python vocab)
	std::vector<double> m_idf;                        // idf[index]


	bool m_useIdf;
	bool m_sublinearTf;

public:

	ChatDetector()
		: m_useIdf(true), m_sublinearTf(false)
	{}
	~ChatDetector() {}

	// ================================
	// Setter Interfaces
	// ================================
	void SetClasses(const std::vector<std::string>& classes)
	{
		m_classes = classes;
	}

	void SetCoef(const std::vector< std::vector<double> >& coef)
	{
		m_coef = coef;
	}

	void SetIntercept(const std::vector<double>& intercept)
	{
		m_intercept = intercept;
	}

	void SetVocab(const std::map<std::string, int>& vocab)
	{
		m_vocab = vocab;
	}

	void SetIdf(const std::vector<double>& idf)
	{
		m_idf = idf;
	}

	void SetUseIdf(bool flag)
	{
		m_useIdf = flag;
	}

	void SetSublinearTf(bool flag)
	{
		m_sublinearTf = flag;
	}

	static void SimpleTokenize(const std::string& text, std::vector<std::string>& outTokens)
	{
		std::string cur;
		for (size_t i = 0; i < text.size(); ++i)
		{
			unsigned char ch = (unsigned char)text[i];
			if (isalnum(ch) || (ch & 0x80))
			{
				cur.push_back(text[i]);
			}
			else
			{
				if (!cur.empty())
				{
					outTokens.push_back(cur);
					cur.clear();
				}
			}
		}
		if (!cur.empty())
			outTokens.push_back(cur);
	}

	std::vector<double> BuildFeature(const std::string& chat) const
	{
		std::vector<double> feature(m_vocab.size(), 0.0);

		std::vector<std::string> tokens;
		SimpleTokenize(chat, tokens);

		if (tokens.empty())
			return feature;

		// 1) TF 계산 (count)
		std::map<int, int> tfCount;
		size_t i;

		for (i = 0; i < tokens.size(); ++i)
		{
			std::map<std::string, int>::const_iterator it = m_vocab.find(tokens[i]);
			if (it != m_vocab.end())
			{
				tfCount[it->second] += 1;
			}
		}


		std::map<int, int>::const_iterator it2;
		for (it2 = tfCount.begin(); it2 != tfCount.end(); ++it2)
		{
			int idx = it2->first;
			int tf = it2->second;
			double val = (double)tf;

			if (m_sublinearTf)
			{
				val = 1.0 + log((double)tf);
			}
			if (m_useIdf && idx < (int)m_idf.size())
			{
				val *= m_idf[idx];
			}

			feature[idx] = val;
		}

		return feature;
	}


	int Predict(const std::vector<double>& x) const
	{
		double bestScore = -1e18;
		int bestClass = -1;

		size_t c;
		for (c = 0; c < m_coef.size(); ++c)
		{
			double score = m_intercept[c];

			size_t i;
			for (i = 0; i < x.size(); ++i)
			{
				score += x[i] * m_coef[c][i];
			}

			if (score > bestScore)
			{
				bestScore = score;
				bestClass = (int)c;
			}
		}

		return bestClass;
	}


	std::string Classify(const std::string& chat) const
	{
		std::vector<double> feat = BuildFeature(chat);
		int idx = Predict(feat);

		if (idx < 0 || idx >= (int)m_classes.size())
			return "Unknown";

		return m_classes[idx];
	}
};

#endif
